import os
import tarfile
from urllib.request import urlretrieve

import numpy as np
from PIL import Image


def download_from_url(url, filename):
    urlretrieve(url, filename)

def extract_tarfile(filename, dest=os.getcwd()):
    with tarfile.open(filename, "r") as f:
        f.extractall(path=dest, filter=lambda tarinfo, path: tarinfo)


def get_formulas(filename):
    with open(filename, 'r') as f:
        formulas = f.readlines()
    return formulas


def get_formula_length_histogram(formulas):
    histogram = {}
    for formula in formulas:
        tokens = formula.split()
        length = len(tokens)
        if length in histogram:
            histogram[length] += 1
        else:
            histogram[length] = 1
    return dict(sorted(histogram.items()))

def get_max_length(formulas, min_occurrences=0):
    histogram = get_formula_length_histogram(formulas)
    max_length = max(histogram.keys())
    for length, occurrences in histogram.items():
        if occurrences > min_occurrences:
            max_length = length
    return max_length

def crop(filename, padding=8):
    # Load image
    image = Image.open(filename).convert("L")
    arr = np.array(image)

    # Identify bounding box of non-white pixels
    non_white_pixels = np.where(arr < 255)
    if len(non_white_pixels[0]) == 0:
        print(f"{filename.name} does not contain any text")
        return None
    y_min, y_max = non_white_pixels[0].min(), non_white_pixels[0].max()
    x_min, x_max = non_white_pixels[1].min(), non_white_pixels[1].max()

    # Crop with padding
    y_min = max(0, y_min - padding)
    y_max = min(image.height, y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(image.width, x_max + padding)
    cropped = arr[y_min:y_max, x_min:x_max]

    # Convert back to PIL image
    cropped_image = Image.fromarray(cropped)
    return cropped_image



class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)


