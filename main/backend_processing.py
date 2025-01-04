from typing import Dict, List, Tuple, Optional, Union
import os
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
from pydub import AudioSegment

def spectral_subtraction(
    audio: np.ndarray, 
    sr: int, 
    noise_start: float = 0, 
    noise_end: float = 1
) -> np.ndarray:
    """
    Cleans up audio by removing background noise using spectral subtraction.
    Takes a sample of background noise from the start of the audio and removes
    similar noise patterns from the entire signal.
    
    Args:
        audio: Raw audio signal
        sr: Sample rate
        noise_start: Start time for noise sample in seconds
        noise_end: End time for noise sample in seconds
    """
    # Get a sample of what we think is background noise
    noise_profile = audio[int(noise_start * sr):int(noise_end * sr)]
    noise_spectrum = np.abs(librosa.stft(noise_profile)).mean(axis=1)
    
    # Convert audio to frequency domain
    S = librosa.stft(audio)
    magnitude, phase = np.abs(S), np.angle(S)
    
    # Subtract noise spectrum and ensure we don't go negative
    reduced_magnitude = np.maximum(magnitude - noise_spectrum[:, None], 0)
    cleaned_audio = librosa.istft(reduced_magnitude * np.exp(1j * phase))
    return cleaned_audio

def bandpass_filter(
    audio: np.ndarray, 
    sr: int, 
    lowcut: int = 300, 
    highcut: int = 3400, 
    order: int = 6
) -> np.ndarray:
    """
    Applies bandpass filter to focus on human voice frequencies.
    Most speech falls between 300-3400 Hz, so we can filter out other stuff.
    
    Args:
        audio: Audio signal to filter
        sr: Sample rate
        lowcut: Lower frequency cutoff
        highcut: Upper frequency cutoff
        order: Filter order (higher = sharper cutoff but more processing)
    """
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, audio)

def trim_silence(
    audio: np.ndarray, 
    top_db: int = 30
) -> np.ndarray:
    """
    Cuts off silence from start and end of audio.
    Uses librosa's trim function to detect quiet parts.
    
    Args:
        audio: Audio to trim
        top_db: Volume threshold to consider as silence
    """
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed_audio

def clean_audio(
    audio: np.ndarray, 
    sr: int
) -> np.ndarray:
    """
    Main audio cleanup pipeline - runs all our cleaning steps in sequence.
    
    Args:
        audio: Raw audio signal
        sr: Sample rate
    """
    cleaned_audio_ss = spectral_subtraction(audio, sr)
    cleaned_audio_bp = bandpass_filter(cleaned_audio_ss, sr)
    cleaned_audio_final = trim_silence(cleaned_audio_bp)
    return cleaned_audio_final

def detect_number_of_speakers(
    audio_path: str, 
    min_segment_length: float = 0.75, 
    energy_threshold: float = 1e-5
) -> int:
    """
    Tries to figure out if we have one or two speakers in the audio.
    Uses voice embeddings to check how different parts of the audio are from each other.
    
    Args:
        audio_path: Path to audio file
        min_segment_length: Minimum length of each audio chunk to analyze
        energy_threshold: Minimum volume to consider as actual speech
    """
    try:
        # Load and preprocess audio
        wav = preprocess_wav(audio_path)
        encoder = VoiceEncoder()

        # Split audio into chunks
        segment_length = int(min_segment_length * 16000)
        segments = [
            wav[i : i + segment_length] 
            for i in range(0, len(wav), segment_length)
            if len(wav[i : i + segment_length]) == segment_length
        ]

        if len(segments) < 2:
            return 1

        # Get voice embeddings for each chunk
        embeddings = np.array([encoder.embed_utterance(seg) for seg in segments])

        # Calculate how similar chunks are to each other
        similarity_matrix = np.zeros((len(embeddings), len(embeddings)))
        for i in range(len(embeddings)):
            for j in range(len(embeddings)):
                similarity_matrix[i][j] = np.dot(embeddings[i], embeddings[j])

        similarity_variance = np.var(similarity_matrix)
        print(f"[DEBUG] Similarity variance: {similarity_variance}")
        
        # If variance is high enough, probably multiple speakers
        VARIANCE_THRESHOLD = 0.015
        if similarity_variance > VARIANCE_THRESHOLD:
            clusterer = SpectralClusterer(min_clusters=2, max_clusters=2)
            labels = clusterer.predict(embeddings)
            
            # Check if each detected speaker actually has enough volume
            cluster_energies = {}
            for lbl in set(labels):
                cluster_segments = [segments[i] for i, lab in enumerate(labels) if lab == lbl]
                cluster_audio = np.concatenate(cluster_segments)
                energy = np.sum(np.square(cluster_audio))
                cluster_energies[lbl] = energy

            non_silent_clusters = [lbl for lbl, eng in cluster_energies.items() if eng > energy_threshold]
            real_speakers = len(non_silent_clusters)
            
            return max(1, real_speakers)
        
        return 1

    except Exception as e:
        print(f"[WARN] Speaker detection hit an error: {e}")
        # If we got far enough to calculate variance, use that as fallback
        if "similarity_variance" in locals() and locals()["similarity_variance"] > 0.015:
            return 2
        return 1

def diarize_audio(
    audio_path: str, 
    min_segment_length: float = 0.75
) -> Dict[int, List[Tuple[float, float]]]:
    """
    Splits audio into segments by speaker. Returns timestamps for each speaker's parts.
    
    Args:
        audio_path: Path to audio file
        min_segment_length: Minimum length of each segment in seconds
    """
    wav = preprocess_wav(audio_path)
    encoder = VoiceEncoder()
    
    # Split audio into chunks
    segment_length = int(min_segment_length * 16000)
    segments = [
        wav[i : i + segment_length] 
        for i in range(0, len(wav), segment_length)
        if len(wav[i : i + segment_length]) == segment_length
    ]
    
    # Get voice embeddings and cluster them
    embeddings = np.array([encoder.embed_utterance(segment) for segment in segments])
    clusterer = SpectralClusterer(min_clusters=2, max_clusters=2)
    labels = clusterer.predict(embeddings)
    
    # Group segments by speaker
    speaker_segments = {label: [] for label in set(labels)}
    current_speaker = labels[0]
    segment_start = 0
    
    for i in range(1, len(labels)):
        if labels[i] != current_speaker:
            speaker_segments[current_speaker].append((segment_start * min_segment_length, i * min_segment_length))
            segment_start = i
            current_speaker = labels[i]
    
    speaker_segments[current_speaker].append((segment_start * min_segment_length, len(labels) * min_segment_length))
    
    # Merge segments that are really close together
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
        
        if current_segment and current_segment[1] - current_segment[0] >= MIN_SEGMENT_DURATION:
            merged_segments.append(tuple(current_segment))
        
        speaker_segments[speaker] = merged_segments
    
    return speaker_segments

def process_audio_file(original_path: str) -> Tuple[str, str]:
    """
    Main processing pipeline - handles everything from separating vocals
    to transcription and getting AI responses.
    
    Args:
        original_path: Path to the original audio file
    
    Returns:
        Tuple of question text and answer text
    """
    # Set up our folder structure
    question_folder = os.path.dirname(os.path.dirname(original_path))
    clean_folder = os.path.join(question_folder, "clean")
    noisy_folder = os.path.join(question_folder, "noisy")
    
    # Use Demucs to pull out just the vocals
    temp_folder = os.path.join(question_folder, "temp")
    os.makedirs(temp_folder, exist_ok=True)
    demucs.separate.main([
        "--mp3", "--two-stems", "vocals", 
        "--device", "cpu", "--out", temp_folder, 
        original_path
    ])
    
    # Move files where we want them
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
    
    # Clean up the vocals
    audio_vocals, sr = librosa.load(vocals_path, sr=None)
    cleaned_audio = clean_audio(audio_vocals, sr)
    cleaned_path = os.path.join(clean_folder, "cleaned_vocals.wav")
    sf.write(cleaned_path, cleaned_audio, sr)
    
    # Figure out how many people are talking
    num_speakers = detect_number_of_speakers(cleaned_path)
    print(f"[INFO] Found {num_speakers} speaker(s)")
    
    # Set up Whisper for transcription
    whisper_model = whisper.load_model("base")
    
    if num_speakers == 1:
        # Simple case - just one person talking
        result = whisper_model.transcribe(cleaned_path)
        question_text = result["text"].strip()
    else:
        # Multiple speakers - need to split it up
        speaker_segments = diarize_audio(cleaned_path)
        
        # Load audio at 16k for consistent indexing
        multi_spk_audio, sr_16k = librosa.load(cleaned_path, sr=16000)
        
        transcripts = {}
        for spk_label, segments in speaker_segments.items():
            total_duration = sum(end - start for start, end in segments)
            
            if total_duration < 0.5:  # Skip tiny segments
                continue
                
            # Pull out this speaker's parts
            speaker_audio = np.zeros_like(multi_spk_audio)
            for start_sec, end_sec in segments:
                start_idx = int(start_sec * sr_16k)
                end_idx = int(end_sec * sr_16k)
                if end_idx <= len(multi_spk_audio):
                    speaker_audio[start_idx:end_idx] = multi_spk_audio[start_idx:end_idx]
            
            # Clean and transcribe just this speaker
            cleaned_speaker_audio = clean_audio(speaker_audio, sr_16k)
            temp_speaker_file = os.path.join(question_folder, f"temp_speaker_{spk_label}.wav")
            sf.write(temp_speaker_file, cleaned_speaker_audio, sr_16k)
            spk_result = whisper_model.transcribe(temp_speaker_file)
            
            transcripts[f"SPEAKER_{spk_label}"] = spk_result["text"].strip()
            os.remove(temp_speaker_file)
        
        question_text = "\n".join(
            f"Speaker {spk}:\n{text}" for spk, text in transcripts.items()
        )
    
    # Get AI to answer the question
    model = OllamaLLM(model="llama3.1")
    
    template = """
    Answer the question to the best of your ability and be concise. Correct any grammar mistakes in the question.
    Question: {question}
    Answer:
    """ if num_speakers == 1 else """
    Given these speaker transcripts, identify the main question or discussion 
    and provide a clear, concise answer. Correct any grammar mistakes in the question.

    {question}

    Answer:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    answer_text = chain.invoke({"question": question_text})
    
    # Save everything to a file
    with open(os.path.join(question_folder, "result.txt"), "w") as f:
        if num_speakers == 1:
            f.write(f"Question: {question_text}\n\nAnswer: {answer_text}\n")
        else:
            f.write("Speaker Segments:\n")
            f.write(question_text)
            f.write(f"\n\nAnswer:\n{answer_text}\n")
    
    return question_text, answer_text

def compress_audio_files(question_folder: str) -> None:
    """
    Converts WAV files to MP3 to save space. 
    
    Args:
        question_folder: Path to the folder containing audio files
    """
    clean_folder = os.path.join(question_folder, "clean")
    noisy_folder = os.path.join(question_folder, "noisy")
    
    for folder in [clean_folder, noisy_folder]:
        for filename in os.listdir(folder):
            if filename.endswith('.wav'):
                wav_path = os.path.join(folder, filename)
                mp3_path = wav_path.replace('.wav', '.mp3')
                
                # Convert to MP3
                audio = AudioSegment.from_wav(wav_path)
                audio.export(mp3_path, format='mp3', bitrate='128k')
                
                # Get rid of the WAV file
                os.remove(wav_path)