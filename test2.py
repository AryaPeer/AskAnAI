import os
import datetime
import whisper
from resemblyzer import preprocess_wav, VoiceEncoder
from pathlib import Path
from spectralcluster import SpectralClusterer
import numpy as np
import librosa
import soundfile as sf
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

def diarize_audio(audio_path, min_segment_length=1.0):
    wav = preprocess_wav(audio_path)
    encoder = VoiceEncoder()
    segment_length = int(min_segment_length * 16000)
    segments = [wav[i:i+segment_length] for i in range(0, len(wav), segment_length) if len(wav[i:i+segment_length]) == segment_length]
    embeddings = np.array([encoder.embed_utterance(segment) for segment in segments])
    clusterer = SpectralClusterer(min_clusters=2, max_clusters=2)
    labels = clusterer.predict(embeddings)
    speaker_segments = {0: [], 1: []}
    for i, label in enumerate(labels):
        start_time = i * min_segment_length
        end_time = (i + 1) * min_segment_length
        speaker_segments[label].append((start_time, end_time))
    return speaker_segments

def transcribe_segments(audio_path, speaker_segments):
    audio, sr = librosa.load(audio_path, sr=16000)
    model = whisper.load_model("base")
    transcripts = {}
    for speaker, segments in speaker_segments.items():
        speaker_audio = np.zeros_like(audio)
        for start, end in segments:
            start_idx = int(start * sr)
            end_idx = int(end * sr)
            if end_idx <= len(audio):
                speaker_audio[start_idx:end_idx] = audio[start_idx:end_idx]
        temp_file = f"temp_speaker_{speaker}.wav"
        sf.write(temp_file, speaker_audio, sr)
        result = model.transcribe(temp_file)
        transcripts[f"SPEAKER_{speaker}"] = result["text"]
        os.remove(temp_file)
    return transcripts

def main():
    load_dotenv()
    model = OllamaLLM(model="llama3.1")
    output_directory = "Questions"
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    question_folder = os.path.join(output_directory, current_time)
    os.makedirs(question_folder, exist_ok=True)
    audio_path = "sample_question.mp3"
    speaker_segments = diarize_audio(audio_path)
    transcripts = transcribe_segments(audio_path, speaker_segments)
    formatted_transcripts = "\n\n".join([
        f"Speaker {speaker}:\n{text}"
        for speaker, text in transcripts.items()
    ])
    template = """
    Given these speaker transcripts, identify the main question and provide a clear answer:
    {transcripts}
    
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    llm_result = chain.invoke({"transcripts": formatted_transcripts})
    transcription_file = os.path.join(question_folder, "transcription.txt")
    with open(transcription_file, "w") as file:
        file.write("Speaker Transcripts:\n")
        file.write(formatted_transcripts)
        file.write("\n\nLLaMA Response:\n")
        file.write(str(llm_result))
    print("Processing complete. Results saved to:", transcription_file)

if __name__ == "__main__":
    main()
