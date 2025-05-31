import whisper
import sounddevice as sd
import numpy as np

model = whisper.load_model("base")

def record(duration=3, samplerate=16000):
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    return np.squeeze(audio)

def transcribe(audio: np.ndarray):
    import scipy.io.wavfile
    scipy.io.wavfile.write("voice.wav", 16000, audio)
    result = model.transcribe("voice.wav", fp16=False, language='English')
    return result['text']


