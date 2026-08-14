import os
import sys
import threading
import time
import traceback
from datetime import datetime

import tkinter as tk
from tkinter import messagebox

from core import Recorder, transcribe

APP_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
OUTPUT_DIR = os.path.join(APP_DIR, "reunioes")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Transcritor de Reuniões")
        self.root.geometry("420x260")
        self.root.resizable(False, False)

        self.recorder = None
        self.recording = False
        self.start_time = None
        self.last_output = None

        self.status_var = tk.StringVar(value="Pronto para gravar")
        self.timer_var = tk.StringVar(value="")

        tk.Label(root, text="Transcritor de Reuniões", font=("Segoe UI", 16, "bold")).pack(pady=(20, 5))
        tk.Label(root, textvariable=self.status_var, font=("Segoe UI", 10), wraplength=380, justify="center").pack(pady=5)
        tk.Label(root, textvariable=self.timer_var, font=("Segoe UI", 22)).pack(pady=5)

        self.main_btn = tk.Button(
            root, text="● Gravar", font=("Segoe UI", 12, "bold"),
            bg="#c0392b", fg="white", width=20, height=2,
            command=self.toggle_recording
        )
        self.main_btn.pack(pady=10)

        self.open_folder_btn = tk.Button(
            root, text="Abrir pasta das reuniões", command=self.open_folder
        )
        self.open_folder_btn.pack(pady=5)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.recorder = Recorder()
        self.recorder.start()
        self.recording = True
        self.start_time = time.time()
        self.main_btn.config(text="■ Parar e Transcrever", bg="#7f8c8d")
        self.status_var.set("Gravando mic + áudio do sistema...")
        self._tick()

        if not self.recorder.loopback_found:
            self.status_var.set("Gravando (aviso: áudio do sistema não encontrado — só mic)")

    def _tick(self):
        if self.recording:
            elapsed = int(time.time() - self.start_time)
            self.timer_var.set(f"{elapsed // 60:02d}:{elapsed % 60:02d}")
            self.root.after(500, self._tick)

    def stop_recording(self):
        self.recording = False
        self.main_btn.config(state="disabled")
        self.status_var.set("Parando gravação...")
        threading.Thread(target=self._process, daemon=True).start()

    def _process(self):
        now = datetime.now().strftime("%Y-%m-%d_%H-%M")
        wav_path = os.path.join(OUTPUT_DIR, f"reuniao_{now}.wav")
        md_path = os.path.join(OUTPUT_DIR, f"reuniao_{now}.md")

        try:
            self._set_status("Mixando áudio...")
            self.recorder.stop_and_save(wav_path)

            def progress(msg):
                self._set_status(msg)

            transcribe(wav_path, md_path, progress=progress)
            os.remove(wav_path)

            self.last_output = md_path
            self._set_status(f"Concluído: {os.path.basename(md_path)}")
        except Exception as e:
            traceback.print_exc()
            self._set_status("Erro na transcrição — veja detalhes.")
            self._show_error(str(e))
        finally:
            self._reset_button()

    def _set_status(self, text):
        self.root.after(0, lambda: self.status_var.set(text))

    def _show_error(self, text):
        self.root.after(0, lambda: messagebox.showerror("Erro", text))

    def _reset_button(self):
        def reset():
            self.main_btn.config(text="● Gravar", bg="#c0392b", state="normal")
            self.timer_var.set("")
        self.root.after(0, reset)

    def open_folder(self):
        os.startfile(OUTPUT_DIR)

    def on_close(self):
        if self.recording:
            if not messagebox.askyesno("Sair", "Uma gravação está em andamento. Sair mesmo assim (a gravação será perdida)?"):
                return
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
