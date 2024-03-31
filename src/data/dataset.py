import os

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from collections import Counter
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab

import torchvision.transforms as transforms


from typing import Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
import json


class Tokenizer:
    def __init__(self, formulas, max_len=150):
        # self.tokenizer = get_tokenizer("basic_english")
        self.tokenizer = get_tokenizer(None)
        self.formulas = formulas
        self.vocab = self._build_vocab(formulas)
        self.vocab.set_default_index(self.vocab['<unk>'])
        self.max_len = max_len
        self.ignore_indices = {self.vocab['<pad>'], self.vocab['<bos>'], self.vocab['<eos>'], self.vocab['<unk>']}

    def __len__(self):
        return len(self.vocab)

    def _build_vocab(self, formulas):
        counter = Counter()
        for formula in formulas:
            counter.update(self.tokenizer(formula))
        return vocab(counter, specials=['<pad>', '<bos>', '<eos>', '<unk>'], min_freq=2)
        # return self.vocab
    
    def encode(self, formula, with_padding=False):
        tokens = self.tokenizer(formula)
        # add the bos and eos to begining and end of the tokens
        tokens = ['<bos>'] + tokens + ['<eos>']
        type(tokens)
        # add padding upto max length if max_len - len(tokens) > 0
        # tokens += ['<pad>'] * (max_len - len(tokens))
        if with_padding:
            tokens = self.pad(tokens, self.max_len)
        return [self.vocab[token] for token in tokens]
    
    def decode(self, tokens):
        # remove the bos and eos tokens
        # tokens = tokens[1:-1]
        if tokens[0] == self.vocab['<bos>']:
            tokens = tokens[1:]
        if tokens[-1] == self.vocab['<eos>']:
            tokens = tokens[:-1]
        return self.vocab.lookup_tokens(tokens)
    
    def pad(self, tokens, max_len):
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
            tokens[-1] = '<eos>'
            return tokens
        return tokens + ['<pad>'] * (max_len - len(tokens))

class BaseDataset(Dataset):
    def __init__(self, dataset_root, images_folder, label_file, data_filter, transform=None):
        self.dataset_root = dataset_root
        self.images_folder = images_folder
        self.label_file = label_file
        self.data_filter = data_filter
        self.transform = transform

        # load the data in self.image_filenames and self.formulas 
        self._load_data(self.data_filter)

        self.tokenizer = Tokenizer(self.formulas)

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
    

class TrainDataset(BaseDataset):
    def __init__(self, dataset_root, images_folder, label_file, data_filter, transform='train'):
        # padding = (0, 0, 224, 224) # pad the right and bottom sides
        if transform == 'train':
            transform = transforms.Compose([
                # transforms.Pad(padding),
                # transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        elif transform == 'test':
            transform = transforms.Compose([
                # transforms.Resize((224, 64)),
                transforms.ToTensor()
            ])
        else:
            raise ValueError("Invalid transform argument. Must be either 'train' or 'test'")
        super().__init__(dataset_root, images_folder, label_file, data_filter, transform)


def get_dataloader(dataset, batch_size=8, num_workers=4):
    # def collate_fn_creator(tokenizer):
    #     def collate_fn(batch):
    #         images, formulas = zip(*batch)
    #         formulas = [tokenizer.encode(formula, with_padding=True) for formula in formulas]
    #         images = torch.stack(images)
    #         formulas = torch.tensor(formulas)
    #         return images, formulas
    #     return collate_fn
    
    def collate_fn_creator(tokenizer):
        def collate_fn(batch):
            images, formulas = zip(*batch)
            formulas = [tokenizer.encode(formula, with_padding=True) for formula in formulas]
            
            # Find max height and width
            max_h = max(img.size(1) for img in images)
            max_w = max(img.size(2) for img in images)
            
            # Pad images
            images = [torch.nn.functional.pad(img, (0, max_w - img.size(2), 0, max_h - img.size(1))) for img in images]
            # Resize images
            # images = [torch.nn.functional.interpolate(img.unsqueeze(0), size=(224, 64)).squeeze(0) for img in images]
            
            images = torch.stack(images)
            formulas = torch.tensor(formulas)
            return images, formulas
        return collate_fn
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers, collate_fn=collate_fn_creator(dataset.tokenizer))
    return dataloader

    



