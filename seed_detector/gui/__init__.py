import tkinter as tk
from tkinter import filedialog

from PIL import Image, ImageTk
from ..defaults import DEFAULT_AREA_THRESHOLD, DEFAULT_N_POLYGON_VERTICES
from ..tools import (
    segment_image2,
    get_contours,
    get_minsize_adaptive2,
    filter_bin_image,
    resample_polygon,
)

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

        self.segmentation_label = tk.Label()
        self.segmentation_label.pack()

        self.image = None
        self.mask = None

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
        bin_image = segment_image2(self.image, k=2)

        min_size = get_minsize_adaptive2(bin_image)
        area_threshold = (
            DEFAULT_AREA_THRESHOLD  # size of holes that will be filled within objects
        )
        bin_image = filter_bin_image(
            bin_image, min_size=min_size, area_threshold=area_threshold
        )

        self.mask = (
            Image.fromarray(np.uint8(bin_image * 255))
            .convert('RGB')
            .resize((256, 256), Image.NEAREST)
        )
        self.mask = ImageTk.PhotoImage(self.mask)
        self.segmentation_label.configure(image=self.mask)
        self.segmentation_label.image = self.mask

    def run(self):
        self.window.mainloop()
