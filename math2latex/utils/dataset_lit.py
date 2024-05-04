
from data import (
    MathToLatexDataset,
    get_dataloader,
    get_formulas,
    Tokenizer,
)

import lightning as L

class LitMathToLatexDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.config = config

        _all_formuals = get_formulas('dataset/im2latex_formulas.norm.processed.lst')
        tokenizer = Tokenizer(_all_formuals)

        tokenizer.save_vocab("dataset/tokenizer_vocab.json")

    def setup(self, stage):

        self.tokenizer = Tokenizer()
        self.tokenizer.load_vocab("dataset/tokenizer_vocab.json")

        self.train_dataset = MathToLatexDataset(
            self.config.train_dataset.root,
            self.config.train_dataset.images_folder,
            self.config.train_dataset.label_file,
            self.config.train_dataset.data_filter,
            transform=self.config.train_dataset.transform,
        )

        self.val_dataset = MathToLatexDataset(
            self.config.val_dataset.root,
            self.config.val_dataset.images_folder,
            self.config.val_dataset.label_file,
            self.config.val_dataset.data_filter,
            transform=self.config.val_dataset.transform,
        )

        self.test_dataset = MathToLatexDataset(
            self.config.test_dataset.root,
            self.config.test_dataset.images_folder,
            self.config.test_dataset.label_file,
            self.config.test_dataset.data_filter,
            transform=self.config.test_dataset.transform,
        )

    def train_dataloader(self):
        _train_dataloader = get_dataloader(
            dataset=self.train_dataset,
            tokenizer=self.tokenizer,
            batch_size=self.config.train_dataloader.batch_size,
            num_workers=self.config.train_dataloader.num_workers,
        )
        return _train_dataloader
    
    def val_dataloader(self):
        _val_dataloader = get_dataloader(
            dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            batch_size=self.config.val_dataloader.batch_size,
            num_workers=self.config.val_dataloader.num_workers,
            shuffle=False
        )
        return _val_dataloader
    
    def test_dataloader(self):
        _test_dataloader = get_dataloader(
            dataset=self.test_dataset,
            tokenizer=self.tokenizer,
            batch_size=self.config.test_dataloader.batch_size,
            num_workers=self.config.test_dataloader.num_workers,
            shuffle=False
        )
        return _test_dataloader
    

        