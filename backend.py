import os
import datetime
import shutil
import demucs.separate
import whisper
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter
from resemblyzer import preprocess_wav, VoiceEncoder
from spectralcluster import SpectralClusterer
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import sounddevice as sd

def record_audio(duration=5, sample_rate=44100):
    """
    Records audio from the default microphone for a specified duration.
    Returns the recorded audio as a NumPy array.
    """
    print("Recording...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # Wait until recording is finished
    return recording.flatten()

def spectral_subtraction(audio, sr, noise_start=0, noise_end=1):
    """
    Performs a simple spectral subtraction to reduce stationary background noise.
    noise_start and noise_end specify which portion of the audio is pure noise.
    """
    noise_profile = audio[int(noise_start * sr):int(noise_end * sr)]
    noise_spectrum = np.abs(librosa.stft(noise_profile)).mean(axis=1)
    S = librosa.stft(audio)
    magnitude, phase = np.abs(S), np.angle(S)
    reduced_magnitude = np.maximum(magnitude - noise_spectrum[:, None], 0)
    cleaned_audio = librosa.istft(reduced_magnitude * np.exp(1j * phase))
    return cleaned_audio

def bandpass_filter(audio, sr, lowcut=300, highcut=3400, order=6):
    """
    Applies a Butterworth bandpass filter between lowcut and highcut frequencies.
    Useful for focusing on typical vocal ranges.
    """
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_audio = lfilter(b, a, audio)
    return filtered_audio

def trim_silence(audio, top_db=30):
    """
    Trims leading and trailing silence from audio using a decibel threshold.
    """
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed_audio

def clean_audio(audio, sr):
    """
    Applies a series of cleaning steps: spectral subtraction, bandpass filtering,
    and silence trimming to produce a cleaner vocal signal.
    """
    cleaned_audio_ss = spectral_subtraction(audio, sr)
    cleaned_audio_bp = bandpass_filter(cleaned_audio_ss, sr)
    cleaned_audio_final = trim_silence(cleaned_audio_bp)
    return cleaned_audio_final

def detect_number_of_speakers(audio_path, min_segment_length=0.75, energy_threshold=1e-5):
    """
    Attempts to detect whether an audio file has one or two speakers using:
      1) Simple variance check on speaker embeddings
      2) Spectral clustering if the variance is above a threshold
    Falls back to 1 speaker if an error occurs or if the variance is low.
    """
    try:
        wav = preprocess_wav(audio_path)
        encoder = VoiceEncoder()

        # Break audio into segments for better speaker embedding coverage
        segment_length = int(min_segment_length * 16000)
        segments = [
            wav[i : i + segment_length] 
            for i in range(0, len(wav), segment_length)
            if len(wav[i : i + segment_length]) == segment_length
        ]

        # If there's too little data, assume 1 speaker
        if len(segments) < 2:
            return 1

        # Compute embeddings for each segment
        embeddings = np.array([encoder.embed_utterance(seg) for seg in segments])

        # Build a similarity matrix
        similarity_matrix = np.zeros((len(embeddings), len(embeddings)))
        for i in range(len(embeddings)):
            for j in range(len(embeddings)):
                similarity_matrix[i][j] = np.dot(embeddings[i], embeddings[j])

        # Check the variance of similarity. Higher variance -> more likely multiple speakers.
        similarity_variance = np.var(similarity_matrix)
        print(f"[DEBUG] Similarity variance: {similarity_variance}")
        
        VARIANCE_THRESHOLD = 0.015
        if similarity_variance > VARIANCE_THRESHOLD:
            # Apply spectral clustering to see if we have 1 or 2 distinct groups
            clusterer = SpectralClusterer(min_clusters=2, max_clusters=2)
            labels = clusterer.predict(embeddings)
            
            # Compute energy for each cluster to see if it's truly active vs. just noise
            cluster_energies = {}
            for lbl in set(labels):
                cluster_segments = [segments[i] for i, lab in enumerate(labels) if lab == lbl]
                cluster_audio = np.concatenate(cluster_segments)
                energy = np.sum(np.square(cluster_audio))
                cluster_energies[lbl] = energy

            # Count how many clusters are non-silent
            non_silent_clusters = [lbl for lbl, eng in cluster_energies.items() if eng > energy_threshold]
            real_speakers = len(non_silent_clusters)
            
            # At least 1 speaker if there's any data
            return max(1, real_speakers)
        
        return 1

    except Exception as e:
        print(f"[WARN] Error in speaker detection: {e}")
        # If we encountered an error but noticed high variance, guess 2 speakers; otherwise, guess 1.
        if "similarity_variance" in locals() and locals()["similarity_variance"] > 0.015:
            return 2
        return 1

def diarize_audio(audio_path, min_segment_length=0.75):
    """
    Uses spectral clustering to label short segments of audio by speaker.
    Returns a dictionary mapping speaker_label to lists of (start_time, end_time) tuples.
    """
    wav = preprocess_wav(audio_path)
    encoder = VoiceEncoder()
    
    segment_length = int(min_segment_length * 16000)
    segments = [
        wav[i : i + segment_length] 
        for i in range(0, len(wav), segment_length)
        if len(wav[i : i + segment_length]) == segment_length
    ]
    
    embeddings = np.array([encoder.embed_utterance(segment) for segment in segments])
    
    clusterer = SpectralClusterer(min_clusters=2, max_clusters=2)
    labels = clusterer.predict(embeddings)
    
    speaker_segments = {label: [] for label in set(labels)}
    current_speaker = labels[0]
    segment_start = 0
    
    # Group segments that share the same label
    for i in range(1, len(labels)):
        if labels[i] != current_speaker:
            speaker_segments[current_speaker].append((segment_start * min_segment_length, i * min_segment_length))
            segment_start = i
            current_speaker = labels[i]
    
    # Append the last group
    speaker_segments[current_speaker].append((segment_start * min_segment_length, len(labels) * min_segment_length))
    
    # Optionally merge very short segments to avoid fragmentation
    MIN_SEGMENT_DURATION = 0.5
    for speaker in speaker_segments:
        merged_segments = []
        current_segment = None
        
        for seg in speaker_segments[speaker]:
            if current_segment is None:
                current_segment = list(seg)
            else:
                if seg[0] - current_segment[1] < MIN_SEGMENT_DURATION:
                    current_segment[1] = seg[1]
                else:
                    if current_segment[1] - current_segment[0] >= MIN_SEGMENT_DURATION:
                        merged_segments.append(tuple(current_segment))
                    current_segment = list(seg)
        
        # Check the final segment
        if current_segment and current_segment[1] - current_segment[0] >= MIN_SEGMENT_DURATION:
            merged_segments.append(tuple(current_segment))
        
        speaker_segments[speaker] = merged_segments
    
    return speaker_segments

def main():
    # Prepare folders to save audio and processing results
    output_directory = "Questions"
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    question_folder = os.path.join(output_directory, current_time)
    clean_folder = os.path.join(question_folder, "clean")
    noisy_folder = os.path.join(question_folder, "noisy")
    os.makedirs(clean_folder, exist_ok=True)
    os.makedirs(noisy_folder, exist_ok=True)

    # Wait for user to press Enter, then record
    print("Press Enter to start recording...")
    input()
    audio_data = record_audio()
    
    # Save the raw recorded audio
    original_path = os.path.join(noisy_folder, "original.wav")
    sf.write(original_path, audio_data, 44100)

    # Use Demucs to isolate vocals from background
    temp_folder = os.path.join(question_folder, "temp")
    os.makedirs(temp_folder, exist_ok=True)
    demucs.separate.main([
        "--mp3", "--two-stems", "vocals", 
        "--device", "cpu", "--out", temp_folder, 
        original_path
    ])

    # Move separated files (vocals, background) into designated folders
    vocals_path = os.path.join(clean_folder, "vocals.mp3")
    background_path = os.path.join(noisy_folder, "background.mp3")
    shutil.move(
        os.path.join(temp_folder, "htdemucs", "original", "vocals.mp3"), 
        vocals_path
    )
    shutil.move(
        os.path.join(temp_folder, "htdemucs", "original", "no_vocals.mp3"), 
        background_path
    )
    shutil.rmtree(temp_folder)

    # Load the vocals and run the cleaning pipeline
    audio_vocals, sr = librosa.load(vocals_path, sr=None)
    cleaned_audio = clean_audio(audio_vocals, sr)
    cleaned_path = os.path.join(clean_folder, "cleaned_vocals.wav")
    sf.write(cleaned_path, cleaned_audio, sr)

    # Check how many speakers are in the cleaned audio
    num_speakers = detect_number_of_speakers(cleaned_path)
    print(f"[INFO] Detected {num_speakers} speaker(s).")

    # Use the Whisper model for transcription
    whisper_model = whisper.load_model("base")

    if num_speakers == 1:
        # Single-speaker transcription
        result = whisper_model.transcribe(cleaned_path)
        question_text = result["text"].strip()
    else:
        # Multi-speaker transcription with diarization
        speaker_segments = diarize_audio(cleaned_path)

        # Load audio at 16k for consistent indexing
        multi_spk_audio, sr_16k = librosa.load(cleaned_path, sr=16000)

        transcripts = {}
        for spk_label, segments in speaker_segments.items():
            total_duration = sum(end - start for start, end in segments)
            
            # Skip short segments that are probably irrelevant
            if total_duration < 0.5:
                continue

            # Collect just this speaker's segments
            speaker_audio = np.zeros_like(multi_spk_audio)
            for start_sec, end_sec in segments:
                start_idx = int(start_sec * sr_16k)
                end_idx = int(end_sec * sr_16k)
                if end_idx <= len(multi_spk_audio):
                    speaker_audio[start_idx:end_idx] = multi_spk_audio[start_idx:end_idx]

            # Optionally re-clean each speaker's segments
            cleaned_speaker_audio = clean_audio(speaker_audio, sr_16k)

            # Transcribe the extracted speaker portion
            temp_speaker_file = os.path.join(question_folder, f"temp_speaker_{spk_label}.wav")
            sf.write(temp_speaker_file, cleaned_speaker_audio, sr_16k)
            spk_result = whisper_model.transcribe(temp_speaker_file)
            
            transcripts[f"SPEAKER_{spk_label}"] = spk_result["text"].strip()
            os.remove(temp_speaker_file)

        # Combine all speaker text into one block for the AI prompt
        question_text = "\n".join(
            f"Speaker {spk}:\n{text}" for spk, text in transcripts.items()
        )

    # Pass the transcribed text to the Ollama model for a response
    model = OllamaLLM(model="llama3.1")
    
    if num_speakers == 1:
        template = """
        Answer the question to the best of your ability and be concise.
        Question: {question}
        Answer:
        """
    else:
        template = """
        Given these speaker transcripts, identify the main question or discussion 
        and provide a clear, concise answer.

        {question}

        Answer:
        """

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    answer_text = chain.invoke({"question": question_text})

    # Write the transcription and the AI's answer to a file
    with open(os.path.join(question_folder, "result.txt"), "w") as f:
        if num_speakers == 1:
            f.write(f"Question: {question_text}\n\nAnswer: {answer_text}\n")
        else:
            f.write("Speaker Segments:\n")
            f.write(question_text)
            f.write(f"\n\nAnswer:\n{answer_text}\n")

    # Print final transcript and answer in the console
    print(f"\n--- TRANSCRIPTION ---\n{question_text}")
    print(f"\n--- ANSWER ---\n{answer_text}")

if __name__ == "__main__":
    main()
