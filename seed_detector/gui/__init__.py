import tkinter as tk
from tkinter import filedialog

from PIL import Image, ImageTk

import numpy as np


def open_function():
    filename = filedialog.askopenfilename(title='open')
    return filename


class GUIApp:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title('Seed Detector')
        self.window.geometry('800x600')
        label = tk.Label(text="Python rocks!")
        label.pack()

        btn = tk.Button(self.window, text='open image', command=self.open_image)
        btn.pack()

        self.image_label = tk.Label()
        self.image_label.pack()

        btn2 = tk.Button(self.window, text='segment', command=self.segment)
        btn2.pack()

        self.image = None

    def open_image(self):
        filename = open_function()
        img = Image.open(filename)
        self.image = np.asanyarray(img.copy())
        img = img.resize((250, 250), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        if self.image_label is None:
            panel = tk.Label(image=img)
            panel.image = img
            panel.pack()
            self.image_label = panel
        else:
            self.image_label.configure(image=img)
            self.image_label.image = img

    def segment(self):
        print(self.image)

    def run(self):
        self.window.mainloop()
