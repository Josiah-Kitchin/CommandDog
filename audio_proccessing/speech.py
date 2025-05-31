import whisper
import sounddevice as sd
import numpy as np

model = whisper.load_model("base")

def record(duration=3, samplerate=16000):
    print("Recording")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    return np.squeeze(audio)

def transcribe(audio: np.ndarray):
    # Save WAV file
    import scipy.io.wavfile
    scipy.io.wavfile.write("temp.wav", 16000, audio)
    result = model.transcribe("temp.wav")
    return result['text']

audio = record()
text = transcribe(audio)
print("You said:", text)


