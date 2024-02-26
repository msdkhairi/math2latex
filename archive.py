import tensorflow as tf
from PIL import Image
import numpy as np
import os

class ImageLatexDataset(tf.data.Dataset):
    def _generator(image_dir, image_list_file, label_file):
        with open(image_list_file, 'r') as f1, open(label_file, 'r') as f2:
            labels = f2.readlines()
            for line in f1:
                image_file, label_index = line.strip().split()
                label_index = int(label_index)
                image_path = os.path.join(image_dir, image_file)
                image = np.array(Image.open(image_path).convert('RGB'))  # Ensure image is RGB
                label = labels[label_index].strip()
                yield (image, label)

    def get_instance(idx):
        return 

    def __new__(cls, image_dir, image_list_file, label_file):
        return tf.data.Dataset.from_generator(
            cls._generator,
            args=(image_dir, image_list_file, label_file),
            output_signature=(
                tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),  # Update shape to match RGB images
                tf.TensorSpec(shape=(), dtype=tf.string)
            )
        )


image_dir = 'dataset/formula_images_processed/'
image_list_file = 'dataset/im2latex_train_filter.lst'
label_file = 'dataset/im2latex_formulas.norm.lst'
dataset = ImageLatexDataset(image_dir, image_list_file, label_file)

# Test the dataset
for image, label in dataset.take(1):  # Take only 1 pair for testing
    print("Image shape:", image.shape)
    print("Label:", label.numpy())

# Example usage:
image_dir = 'dataset/formula_images_processed/'
image_list_file = 'dataset/im2latex_train_filter.lst'
label_file = 'dataset/im2latex_formulas.norm.lst'
batch_size = 1
image_size = (224, 224)

custom_dataset = CustomDataset(image_dir, image_list_file, label_file, image_size=image_size, batch_size=batch_size)














import os

def list_files_in_directory(directory):
    return os.listdir(directory)

dataset_dir = 'dataset/formula_images_processed/'
all_files = list_files_in_directory(dataset_dir)


from PIL import Image

dataset_dit = "dataset/formula_images_processed/"
image_dir = os.path.join(dataset_dir, all_files[10])
img = Image.open(image_dir)
print(img.size)
img.show()


import os
from PIL import Image

def create_dataset(image_dir, image_list_file, label_file):
    dataset = []
    with open(image_list_file, 'r') as f1, open(label_file, 'r') as f2:
        labels = f2.readlines()
        for line in f1:
            image_file, label_index = line.strip().split()
            label_index = int(label_index)
            image_path = os.path.join(image_dir, image_file)
            image = Image.open(image_path)
            label = labels[label_index].strip()
            dataset.append((image, label))
    return dataset

image_dir = 'dataset/formula_images_processed/'
image_list_file = 'dataset/im2latex_train_filter.lst'
label_file = 'dataset/im2latex_formulas.norm.lst'
dataset = create_dataset(image_dir, image_list_file, label_file)