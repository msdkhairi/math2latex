class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, Config(value) if isinstance(value, dict) else value)

all_configurations = {
    'train_dataset': {
        'root': '/scratch/mka267/workspace-scratch/math2latex-data/dataset',
        'images_folder': 'formula_images_processed',
        'label_file': 'im2latex_formulas.norm.processed.lst',
        'data_filter': 'im2latex_train_filter.lst',
        'transform': 'train'
    },
    'val_dataset': {
        'root': '/scratch/mka267/workspace-scratch/math2latex-data/dataset',
        'images_folder': 'formula_images_processed',
        'label_file': 'im2latex_formulas.norm.processed.lst',
        'data_filter': 'im2latex_validate_filter.lst',
        'transform': 'test'
    },
    'test_dataset': {
        'root': '/scratch/mka267/workspace-scratch/math2latex-data/dataset',
        'images_folder': 'formula_images_processed',
        'label_file': 'im2latex_formulas.norm.processed.lst',
        'data_filter': 'im2latex_test_filter.lst',
        'transform': 'test'
    },
    'train_dataloader': {
        'batch_size': 64,
        'num_workers': 32
    },
    'val_dataloader': {
        'batch_size': 64,
        'num_workers': 32
    },
    'test_dataloader': {
        'batch_size': 64,
        'num_workers': 32
    },
    'model': {
        'd_model': 128,
        'num_heads': 4,
        'num_decoder_layers': 3,
        'dim_feedforward': 256,
        'dropout': 0.3,
        'num_classes': 462,
        'max_len': 150
    },
    'optimizer': {
        'lr': 0.001,
        'weight_decay': 0.0001,
        'milestones': [10],
        'gamma': 0.5
    },
    'trainer': {
        'epochs': 2,
        'gpus': 1,
        'accelerator': 'gpu',
        'batch_size': 32,
        'learning_rate' : 0.001
    }
}

# Create the main config object
config = Config(all_configurations)
