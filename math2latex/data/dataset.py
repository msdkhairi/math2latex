import os
import random
import json

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from collections import Counter
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab

import torchvision.transforms as transforms

from .utils import get_formulas


class Tokenizer:
    def __init__(self, formulas=None, max_len=150):
        # self.tokenizer = get_tokenizer(None)
        self.tokenizer = get_tokenizer("basic_english")
        self.max_len = max_len
        
        if formulas is not None:
            self.vocab = self._build_vocab(formulas)
            self.vocab.set_default_index(self.vocab['<unk>'])
            self.pad_index = self.vocab['<pad>']
            self.ignore_indices = {self.vocab['<pad>'], self.vocab['<bos>'], self.vocab['<eos>'], self.vocab['<unk>']}
        else:
            self.vocab = None

    def _build_vocab(self, formulas):
        counter = Counter()
        for formula in formulas:
            counter.update(self.tokenizer(formula))
        return vocab(counter, specials=['<pad>', '<bos>', '<eos>', '<unk>'], min_freq=2)
    
    def encode(self, formula, with_padding=False):
        tokens = self.tokenizer(formula)
        tokens = ['<bos>'] + tokens + ['<eos>']
        if with_padding:
            tokens = self.pad(tokens, self.max_len)
        # add the bos and eos to begining and end of the tokens
        return [self.vocab[token] for token in tokens]
    
    def decode(self, indices):
        return self.vocab.lookup_tokens(list(indices))
    
    def decode_clean(self, indices):
        # removes the ignore indices from the decoded tokens
        cleaned_indices = [index for index in indices if int(index) not in self.ignore_indices]
        # if self.vocab['<eos>'] in cleaned_indices:
        #     cleaned_indices = cleaned_indices[:cleaned_indices.index(self.vocab['<eos>'])]
        return self.vocab.lookup_tokens(cleaned_indices)
    
    def decode_to_string(self, tokens):
        # returns the decoded tokens as a string
        decoded = self.decode_clean(tokens)
        return ' '.join(decoded)


    def pad(self, tokens, max_len):
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
            tokens[-1] = '<eos>'
            return tokens
        return tokens + ['<pad>'] * (max_len - len(tokens))

    def save_vocab(self, file_path="dataset/tokenizer_vocab.json"):
        # Save the list of tokens which reflects both `itos` and `stoi`
        vocab_data = {
            'itos': self.vocab.get_itos()
        }
        with open(file_path, 'w') as f:
            json.dump(vocab_data, f)

    def load_vocab(self, file_path):
        with open(file_path, 'r') as f:
            vocab_data = json.load(f)
        # Reconstruct the vocabulary from the itos list
        ordered_tokens = vocab_data['itos']
        # Reconstruct the counter from the ordered list
        counter = Counter({token: idx + 1 for idx, token in enumerate(ordered_tokens)})  # idx+1 to ensure non-zero freq
        self.vocab = vocab(counter, specials=['<pad>', '<bos>', '<eos>', '<unk>'])
        self.vocab.set_default_index(self.vocab['<unk>'])
        self.pad_index = self.vocab['<pad>']
        self.ignore_indices = {self.vocab['<pad>'], self.vocab['<bos>'], self.vocab['<eos>'], self.vocab['<unk>']}


    def __len__(self):
        return len(self.vocab)

class BaseDataset(Dataset):
    def __init__(self, dataset_root, images_folder, label_file, data_filter, transform=None):
        self.dataset_root = dataset_root
        self.images_folder = images_folder
        self.label_file = label_file
        self.data_filter = data_filter
        self.transform = transform

        # load the data in self.image_filenames and self.formulas 
        self._load_data(self.data_filter)

        # self.tokenizer = Tokenizer(self.formulas)

    def _load_data(self, data_filter=None):
        # Read the formulas
        with open(os.path.join(self.dataset_root, self.label_file), 'r') as f:
                self.formulas = [line.strip() for line in f]


        if data_filter is not None:
            # Read the image filenames and corresponding label indices
            with open(os.path.join(self.dataset_root, self.data_filter), 'r') as f:
                lines = [line.strip().split() for line in f]
                self.image_filenames = [os.path.join(self.dataset_root, self.images_folder, line[0]) for line in lines]
                self.formulas = [self.formulas[int(line[1])] for line in lines]

        else:
            self.image_filenames = [os.path.join(self.dataset_root, self.images_folder, img) for img in os.listdir(os.path.join(self.dataset_root, self.images_folder))]
            self.formulas = self.formulas[:len(self.image_filenames)]

        assert len(self.image_filenames) == len(self.formulas)

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        formula = self.formulas[idx]

        image = Image.open(image_filename)
        if self.transform:
            image = self.transform(image)

        return image, formula
    

class MathToLatexDataset(BaseDataset):
    def __init__(self, dataset_root, images_folder, label_file, data_filter, transform='train'):
        if transform == 'train':
            transform = transforms.Compose([
                transforms.RandomApply([transforms.RandomAffine(degrees=(-1, 1), scale=(0.6, 1.0), fill=255)], p=0.5),
                transforms.RandomApply([transforms.Lambda(lambda x: self.gaussian_noise(x))], p=0.5),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=1, sigma=0.5)], p=0.5),
                transforms.ToTensor(),
            ])
        elif transform == 'test':
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            raise ValueError("Invalid transform argument. Must be either 'train' or 'test'")
        super().__init__(dataset_root, images_folder, label_file, data_filter, transform)

    def gaussian_noise(self, tensor, mean=0, var_range=(10, 50)):
        var = random.uniform(var_range[0], var_range[1])

        # Convert PIL image to numpy array
        tensor_array = np.array(tensor)
        
        # Generate Gaussian noise
        noise = np.random.normal(mean, var, tensor_array.shape)
        
        # Add noise to the image
        noisy_array = tensor_array + noise
        
        # Clip the values to [0, 255]
        noisy_array = np.clip(noisy_array, 0, 255)
        
        # Convert numpy array back to PIL image
        noisy_tensor = Image.fromarray(np.uint8(noisy_array))
        
        return noisy_tensor


def get_dataloader(dataset, tokenizer=None, batch_size=8, num_workers=4, shuffle=True):
    # if tokenizer is None:
    #     all_formuals = get_formulas('dataset/im2latex_formulas.norm.processed.lst')
    #     tokenizer = Tokenizer(all_formuals)
    def collate_fn_creator(tokenizer):
        def collate_fn(batch):
            images, formulas = zip(*batch)
            formulas = [tokenizer.encode(formula, with_padding=True) for formula in formulas]
            
            # Find max height and width
            max_h = max(img.size(1) for img in images)
            max_w = max(img.size(2) for img in images)
            
            # Pad images
            images = [1. - torch.nn.functional.pad(img, (0, max_w - img.size(2), 0, max_h - img.size(1)), value=1) for img in images]
            
            images = torch.stack(images)
            formulas = torch.tensor(formulas)
            return images, formulas
        return collate_fn
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                            num_workers=num_workers, collate_fn=collate_fn_creator(tokenizer))
    return dataloader

    



