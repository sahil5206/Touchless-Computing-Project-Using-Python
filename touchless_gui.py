import customtkinter as ctk
import subprocess

# Configure appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Create main window
app = ctk.CTk()
app.title("✨ Touchless Control Center")
app.geometry("500x650")
app.resizable(False, False)

# Track the running process
current_process = None

def run_script(script_name):
    global current_process
    try:
        # Stop previous process if still running
        stop_script()

        current_process = subprocess.Popen(["python", script_name])
    except Exception as e:
        print(f"Error: {e}")

def stop_script():
    global current_process
    if current_process and current_process.poll() is None:
        current_process.terminate()
        print("Process terminated.")
        current_process = None

# Heading
title_label = ctk.CTkLabel(app, text="Touchless Control Center", font=ctk.CTkFont(size=24, weight="bold"))
title_label.pack(pady=30)

# Buttons
buttons = [
    ("🖐️ Hand Gesture Keyboard", "virtualkeyboard.py"),
    ("🖱️ Hand Gesture Mouse", "mouse.py"),
    ("👁️ Eye Gesture Mouse", "eye_controlled_mouse.py"),
    ("🔊 Volume Controller", "Gesture_Volume_Control.py"),
    ("🎙️ Voice Assistant", "advanced_voice_assistant.py"),
]

for label, script in buttons:
    btn = ctk.CTkButton(app, text=label, command=lambda s=script: run_script(s), width=300, height=50, corner_radius=10, font=ctk.CTkFont(size=16))
    btn.pack(pady=10)

# Stop Button
stop_btn = ctk.CTkButton(app, text="🛑 Stop Current Process", command=stop_script, width=200, height=40, fg_color="#f57c00", hover_color="#ef6c00")
stop_btn.pack(pady=20)

# Exit Button
exit_btn = ctk.CTkButton(app, text="❌ Exit", command=app.destroy, width=150, height=40, fg_color="#c62828", hover_color="#b71c1c")
exit_btn.pack(pady=10)

app.mainloop()
