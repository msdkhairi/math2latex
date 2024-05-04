import os
import tarfile
from urllib.request import urlretrieve
import re
from tqdm import tqdm
from multiprocessing import Pool

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
        # return a white image
        return Image.new('L', (128, 64), 255)
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


def replace_similar_tokens(line):
    # Define the regular expressions for replacements
    patterns = [
        (r'\\left\(', r'('),
        (r'\\right\)', r')'),
        (r'\\left\[', r'['),
        (r'\\right\]', r']'),
        (r'\\left\{', r'{'),
        (r'\\right\}', r'}'),
        (r'\\vspace(\*)?\{[0-9a-zA-Z.~-]*[^}]*\}', ''),
        (r'\\hspace(\*)?\{[0-9a-zA-Z.~-]*[^}]*\}', '')
    ]

    # Apply replacements
    for pattern, replacement in patterns:
        line = re.sub(pattern, replacement, line)

    return line

def find_and_replace(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            cleaned_line = replace_similar_tokens(line)
            f_out.write(cleaned_line)


def process_image(args):
    filename, dataset_dir, processed_imgs_dir = args
    cropped_image = crop(os.path.join(dataset_dir, filename))
    if cropped_image is not None:
        # Save the cropped image
        cropped_image.save(os.path.join(processed_imgs_dir, filename))
    else:
        print(f"\n{filename} does not contain any text")

def process_images(dataset_dir, processed_imgs_dir):
    # check if the processed images directory exists
    if not os.path.exists(processed_imgs_dir):
        os.makedirs(processed_imgs_dir)
    
    # Get a list of all files in the dataset directory
    img_file_list = [filename for filename in os.listdir(dataset_dir) if filename.endswith('.png')]
    
    print('Processing images...')
    args_list = [(filename, dataset_dir, processed_imgs_dir) for filename in img_file_list]
    with Pool() as pool:
        # Use tqdm to track progress
        with tqdm(total=len(img_file_list), desc='Processing images') as progress_bar:
            for _ in pool.imap_unordered(process_image, args_list):
                progress_bar.update()

    

class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)
