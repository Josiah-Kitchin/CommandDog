

import speech

def main(): 

    print("Recording...")
    audio = speech.record(duration=3)
    text = speech.transcribe(audio)

    print(f"You said: ${text}")



if __name__ == "__main__": 
    main()
