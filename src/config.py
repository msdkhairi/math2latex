train_dataset = {
    'root': 'dataset',
    'images_folder': 'formula_images_processed',
    'label_file': 'im2latex_formulas.norm.lst',
    'data_filter': 'im2latex_train_filter.lst',
    'transform': 'train',
    'batch_size': 8,
    'num_workers': 4
}

val_dataset = {
    'root': 'dataset',
    'images_folder': 'formula_images_processed',
    'label_file': 'im2latex_formulas.norm.lst',
    'data_filter': 'im2latex_validate_filter.lst',
    'transform': 'train',
    'batch_size': 8,
    'num_workers': 4
}

train_dataloader = {
    'batch_size': 8,
    'num_workers': 4
}

val_dataloader = {
    'batch_size': 8,
    'num_workers': 4
}



model = {
    'd_model': 128,
    'dim_feedforward': 256,
    'num_heads': 4,
    'dropout': 0.2,
    'num_decoder_layers': 3,
    'num_classes': 500,
    'max_len': 150
}

optimizer = {
    'lr': 0.001,
    'milestones': [10],
    'gamma': 0.5
}


training = {
    'epochs': 10
}

class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

config = Config({
    'train_dataset': Config(train_dataset),
    'val_dataset': Config(val_dataset),
    'train_dataloader': Config(train_dataloader),
    'val_dataloader': Config(val_dataloader),
    'model': Config(model),
    'optimizer': Config(optimizer),
    'training': Config(training)
})