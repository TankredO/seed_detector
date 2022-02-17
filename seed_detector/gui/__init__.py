import tkinter as tk
from tkinter import filedialog

from PIL import Image, ImageTk


def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename


def open_img():
    x = openfn()
    img = Image.open(x)
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = tk.Label(image=img)
    panel.image = img
    panel.pack()


class GUIApp:
    def __init__(self):
        self.window = tk.Tk()
        label = tk.Label(text="Python rocks!")
        label.pack()

        btn = tk.Button(self.window, text='open image', command=open_img)
        btn.pack()

    def run(self):
        self.window.mainloop()
