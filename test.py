import os
import datetime
import shutil
import demucs.separate
import whisper

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

model = whisper.load_model("base")

result = model.transcribe(vocals_dest)

print(result['text'])

transcription_file = os.path.join(question_folder, "transcription.txt")
with open(transcription_file, "w") as file:
    file.write(result['text'])

print(f"Files saved to folder: {question_folder}")