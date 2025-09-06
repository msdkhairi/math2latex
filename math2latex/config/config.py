
from .utils import Config


dataset_dir = 'dataset'
batch_size = 32
num_workers = 32

all_configurations = {
    'datamodule': {
        'train_dataset_root': dataset_dir,
        'train_dataset_images_folder': 'formula_images_processed',
        'train_dataset_label_file': 'im2latex_formulas.norm.processed.lst',
        'train_dataset_data_filter': 'im2latex_train_filter.lst',
        'train_dataset_transform': 'train',
        'val_dataset_root': dataset_dir,
        'val_dataset_images_folder': 'formula_images_processed',
        'val_dataset_label_file': 'im2latex_formulas.norm.processed.lst',
        'val_dataset_data_filter': 'im2latex_validate_filter.lst',
        'val_dataset_transform': 'test',
        'test_dataset_root': dataset_dir,
        'test_dataset_images_folder': 'formula_images_processed',
        'test_dataset_label_file': 'im2latex_formulas.norm.processed.lst',
        'test_dataset_data_filter': 'im2latex_test_filter.lst',
        'test_dataset_transform': 'test',
        'train_dataloader_batch_size': batch_size,
        'train_dataloader_num_workers': num_workers,
        'val_dataloader_batch_size': batch_size,
        'val_dataloader_num_workers': num_workers,
        'test_dataloader_batch_size': batch_size,
        'test_dataloader_num_workers': num_workers,
    },
    'litmodel': {
        'd_model': 256,
        'num_heads': 8,
        'num_decoder_layers': 6,
        'dim_feedforward': 256,
        'dropout': 0.3,
        'num_classes': 462,
        'max_len': 150,
        'lr': 0.001,
        'weight_decay': 0.0001,
        'milestones': [10],
        'gamma': 0.5
    },
    'trainer': {
        'max_epochs': 201,
        'accelerator': 'auto',
        'strategy': 'auto',
        'enable_progress_bar': True,
        'check_val_every_n_epoch': 10,
    }
}

# Create the main config object
config = Config(all_configurations)
