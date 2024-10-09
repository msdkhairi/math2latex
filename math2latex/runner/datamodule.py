import os

from data import (
    MathToLatexDataset,
    get_dataloader,
    get_formulas,
    Tokenizer,
)

import lightning as L

class LitMathToLatexDataModule(L.LightningDataModule):
    def __init__(self,
                #  train_dataset params
                train_dataset_root: str = 'dataset',
                train_dataset_images_folder: str = 'formula_images_processed',
                train_dataset_label_file: str = 'im2latex_formulas.norm.processed.lst',
                train_dataset_data_filter: str = 'im2latex_train_filter.lst',
                train_dataset_transform: str = 'train',
                # val_dataset params
                val_dataset_root: str = 'dataset',
                val_dataset_images_folder: str = 'formula_images_processed',
                val_dataset_label_file: str = 'im2latex_formulas.norm.processed.lst',
                val_dataset_data_filter: str = 'im2latex_validate_filter.lst',
                val_dataset_transform: str = 'test',
                # test_dataset params
                test_dataset_root: str = 'dataset',
                test_dataset_images_folder: str = 'formula_images_processed',
                test_dataset_label_file: str = 'im2latex_formulas.norm.processed.lst',
                test_dataset_data_filter: str = 'im2latex_test_filter.lst',
                test_dataset_transform: str = 'test',
                # train_dataloader params
                train_dataloader_batch_size: int = 64,
                train_dataloader_num_workers: int = 32,
                # val_dataloader params
                val_dataloader_batch_size: int = 64,
                val_dataloader_num_workers: int = 32,
                # test_dataloader params
                test_dataloader_batch_size: int = 64,
                test_dataloader_num_workers: int = 32,
                ):
        super().__init__()
        self.train_dataset_root = train_dataset_root
        self.train_dataset_images_folder = train_dataset_images_folder
        self.train_dataset_label_file = train_dataset_label_file
        self.train_dataset_data_filter = train_dataset_data_filter
        self.train_dataset_transform = train_dataset_transform

        self.val_dataset_root = val_dataset_root
        self.val_dataset_images_folder = val_dataset_images_folder
        self.val_dataset_label_file = val_dataset_label_file
        self.val_dataset_data_filter = val_dataset_data_filter
        self.val_dataset_transform = val_dataset_transform

        self.test_dataset_root = test_dataset_root
        self.test_dataset_images_folder = test_dataset_images_folder
        self.test_dataset_label_file = test_dataset_label_file
        self.test_dataset_data_filter = test_dataset_data_filter
        self.test_dataset_transform = test_dataset_transform

        self.train_dataloader_batch_size = train_dataloader_batch_size
        self.train_dataloader_num_workers = train_dataloader_num_workers

        self.val_dataloader_batch_size = val_dataloader_batch_size
        self.val_dataloader_num_workers = val_dataloader_num_workers

        self.test_dataloader_batch_size = test_dataloader_batch_size
        self.test_dataloader_num_workers = test_dataloader_num_workers

        formulas_filename = os.path.join(train_dataset_root, train_dataset_label_file)
        all_formuals = get_formulas(formulas_filename)
        tokenizer = Tokenizer(all_formuals)

        tokenizer.save_vocab("tokenizer_vocab.json")

    def setup(self, stage):

        self.tokenizer = Tokenizer()
        self.tokenizer.load_vocab("tokenizer_vocab.json")

        self.train_dataset = MathToLatexDataset(
            self.train_dataset_root,
            self.train_dataset_images_folder,
            self.train_dataset_label_file,
            self.train_dataset_data_filter,
            transform=self.train_dataset_transform,
        )

        self.val_dataset = MathToLatexDataset(
            self.val_dataset_root,
            self.val_dataset_images_folder,
            self.val_dataset_label_file,
            self.val_dataset_data_filter,
            transform=self.val_dataset_transform,
        )

        self.test_dataset = MathToLatexDataset(
            self.test_dataset_root,
            self.test_dataset_images_folder,
            self.test_dataset_label_file,
            self.test_dataset_data_filter,
            transform=self.test_dataset_transform,
        )

    def train_dataloader(self):
        _train_dataloader = get_dataloader(
            dataset=self.train_dataset,
            tokenizer=self.tokenizer,
            batch_size=self.train_dataloader_batch_size,
            num_workers=self.train_dataloader_num_workers,
        )
        return _train_dataloader
    
    def val_dataloader(self):
        _val_dataloader = get_dataloader(
            dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            batch_size=self.val_dataloader_batch_size,
            num_workers=self.val_dataloader_num_workers,
            shuffle=False
        )
        return _val_dataloader
    
    def test_dataloader(self):
        _test_dataloader = get_dataloader(
            dataset=self.test_dataset,
            tokenizer=self.tokenizer,
            batch_size=self.test_dataloader_batch_size,
            num_workers=self.test_dataloader_num_workers,
            shuffle=False
        )
        return _test_dataloader
        