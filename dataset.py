import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.utils import Sequence
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.image import load_img, img_to_array
from keras.utils import Sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class TokenizerWrap(Tokenizer):
    def __init__(self, texts, max_len, reverse=False, num_words=None):
        Tokenizer.__init__(self, num_words=num_words, oov_token='<UNK>')
        self.fit_on_texts(texts)
        self.index_to_word = dict(zip(self.word_index.values(), self.word_index.keys()))
        self.tokens = self.texts_to_sequences(texts)
        self.tokens = pad_sequences(self.tokens, maxlen=max_len, padding='post')
        self.num_words = len(self.word_index) + 1
        self.reverse = reverse

    def caption_to_tokens(self, caption):
        tokens = self.texts_to_sequences([caption])
        return pad_sequences(tokens, padding='post')

    def tokens_to_caption(self, tokens):
        if self.reverse:
            return ' '.join([self.index_to_word[index] for index in tokens[0] if index != 0])
        else:
            return ' '.join([self.index_to_word[index] for index in tokens[0] if index != 0][::-1])

class BaseDataset(Sequence):
    def __init__(self, image_dir, image_list_file, label_file, image_size=(224, 224), batch_size=32, max_sequence_length=150):
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.image_size = image_size 
        self.max_sequence_length = max_sequence_length

        # Read image list and labels
        with open(image_list_file, 'r') as f:
            self.image_list = f.readlines()

        with open(label_file, 'r') as f:
            self.labels = f.readlines()

        # Extract image filenames and labels
        self.filenames = []
        self.labels_list = []
        for line in self.image_list:
            parts = line.strip().split()
            self.filenames.append(parts[0])
            self.labels_list.append(parts[1:])

        # # use tokenizerwrap to tokenize and pad labels
        # self.tokenizer = TokenizerWrap(self.labels, padding='post')
        # self.vocab_size = self.tokenizer.num_words
        
        # Tokenize and pad labels
        # self.tokenizer = Tokenizer(oov_token='<OOV>')
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.labels)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.labels_sequences = self.tokenizer.texts_to_sequences(self.labels)
        self.labels_padded = pad_sequences(self.labels_sequences, maxlen=self.max_sequence_length, padding='post')

    def __len__(self):
        return len(self.filenames) // self.batch_size

    # def __getitem__(self, idx):
    #     batch_x = []
    #     batch_y = []
    #     start_index = idx * self.batch_size
    #     end_index = (idx + 1) * self.batch_size

    #     for i in range(start_index, end_index):
    #         filename = os.path.join(self.image_dir, self.filenames[i])
    #         image = load_img(filename, target_size=self.image_size)
    #         image = img_to_array(image)
    #         image = image / 255.0  # Normalize to [0, 1]
    #         label = self.labels_padded[i]
    #         batch_x.append(image)
    #         batch_y.append(label)

    #     return np.array(batch_x), np.array(batch_y)
    
    def __getitem__(self, idx):
        start_index = idx * self.batch_size
        end_index = (idx + 1) * self.batch_size

        # Get a slice of filenames
        filenames_slice = self.filenames[start_index:end_index]

        # Load and process images in one go using vectorized operations
        image_paths = [os.path.join(self.image_dir, filename) for filename in filenames_slice]
        images = np.stack([img_to_array(load_img(path, target_size=self.image_size)) / 255.0 for path in image_paths])

        # Get labels 
        labels = self.labels_padded[start_index:end_index]

        return images, labels