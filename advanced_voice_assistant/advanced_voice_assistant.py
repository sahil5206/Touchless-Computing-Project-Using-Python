import speech_recognition as sr
import pyttsx3
import pyautogui
import webbrowser
import os
import difflib
import time

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 180)

def speak(text):
    print("[Assistant]:", text)
    engine.say(text)
    engine.runAndWait()

# Command Map
COMMANDS = {
    "open notepad": lambda: os.system("notepad.exe"),
    "open browser": lambda: webbrowser.open("https://www.google.com"),
    "open youtube": lambda: webbrowser.open("https://www.youtube.com"),
    "open our github profile": lambda: webbrowser.open("https://github.com/sahil5206"),
    "open our github repository": lambda: webbrowser.open("https://github.com/sahil5206/Touchless-Computing-Project-Using-Python"),
    "play my favourite song": lambda: webbrowser.open("https://www.youtube.com/watch?v=WPoiaEeoO6s&list=RDWPoiaEeoO6s&start_radio=1"),
    "volume up": lambda: [pyautogui.press("volumeup") for _ in range(5)],
    "volume down": lambda: [pyautogui.press("volumedown") for _ in range(5)],
    "mute": lambda: pyautogui.press("volumemute"),
    "scroll down": lambda: pyautogui.scroll(-1000),
    "scroll up": lambda: pyautogui.scroll(1000),
    "type hello": lambda: pyautogui.write("hello"),
    "exit": lambda: speak("Goodbye!") or exit()
}

def recognize_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        speak("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=7)
            text = recognizer.recognize_google(audio).lower()
            print("Recognized:", text)
            return text
        except sr.WaitTimeoutError:
            speak("I did not hear anything. Try again.")
        except sr.UnknownValueError:
            speak("Sorry, I did not understand that.")
        except sr.RequestError as e:
            speak(f"Could not request results; {e}")
        except Exception as e:
            speak(f"An error occurred: {e}")
    return ""

def match_command(text):
    all_commands = list(COMMANDS.keys())
    best_match = difflib.get_close_matches(text, all_commands, n=1, cutoff=0.5)
    if best_match:
        return best_match[0]
    return None

def main():
    speak("Assistant activated. Say a command.")
    while True:
        command = recognize_command()
        if not command:
            continue

        if command.startswith("type"):
            to_type = command.replace("type", "").strip()
            pyautogui.write(to_type)
            speak(f"Typed: {to_type}")
            continue

        matched = match_command(command)
        if matched:
            speak(f"Executing: {matched}")
            COMMANDS[matched]()
        else:
            speak("Sorry, I do not recognize that command.")

if __name__ == "__main__":
    main()