# audio_module.py

import whisper

def transcribe_audio_file(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result['text']

def transcribe_audio():
    audio_path = input("Digite o caminho para o arquivo de áudio: ")
    transcription = transcribe_audio_file(audio_path)
    print("\nTranscrição do áudio:\n")
    print(transcription)
