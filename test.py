import os
import datetime
import shutil
import demucs.separate
import whisper
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

def spectral_subtraction(audio, sr, noise_start=0, noise_end=1):
    noise_profile = audio[int(noise_start * sr):int(noise_end * sr)]
    noise_spectrum = np.abs(librosa.stft(noise_profile)).mean(axis=1)
    S = librosa.stft(audio)
    magnitude, phase = np.abs(S), np.angle(S)
    reduced_magnitude = np.maximum(magnitude - noise_spectrum[:, None], 0)
    cleaned_audio = librosa.istft(reduced_magnitude * np.exp(1j * phase))
    return cleaned_audio

def bandpass_filter(audio, sr, lowcut=300, highcut=3400, order=6):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_audio = lfilter(b, a, audio)
    return filtered_audio

def trim_silence(audio, top_db=30):
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed_audio

output_directory = "Questions"
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
question_folder = os.path.join(output_directory, current_time)
os.makedirs(question_folder, exist_ok=True)

temp_folder = os.path.join(question_folder, "temp")
os.makedirs(temp_folder, exist_ok=True)

demucs.separate.main(["--mp3", "--two-stems", "vocals", "--device", "cpu", "--out", temp_folder, "sample_question.mp3"])

vocals_src = os.path.join(temp_folder, "htdemucs", "sample_question", "vocals.mp3")
accompaniment_src = os.path.join(temp_folder, "htdemucs", "sample_question", "no_vocals.mp3")

vocals_dest = os.path.join(question_folder, "vocals.mp3")
accompaniment_dest = os.path.join(question_folder, "no_vocals.mp3")

shutil.move(vocals_src, vocals_dest)
shutil.move(accompaniment_src, accompaniment_dest)

shutil.rmtree(temp_folder)

audio, sr = librosa.load(vocals_dest, sr=None)
cleaned_audio_ss = spectral_subtraction(audio, sr, noise_start=0, noise_end=1)
cleaned_audio_bp = bandpass_filter(cleaned_audio_ss, sr)
cleaned_audio_final = trim_silence(cleaned_audio_bp)

cleaned_audio_path = os.path.join(question_folder, "vocals_cleaned.wav")
sf.write(cleaned_audio_path, cleaned_audio_final, sr)

model = whisper.load_model("base")
result = model.transcribe(cleaned_audio_path)

transcription_file = os.path.join(question_folder, "transcription.txt")
with open(transcription_file, "w") as file:
    file.write(result['text'])

print(f"Transcription: {result['text']}")
print(f"Files saved to folder: {question_folder}")