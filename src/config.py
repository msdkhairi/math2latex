train_dataset = {
    'root': 'dataset',
    'images_folder': 'formula_images_processed',
    'label_file': 'im2latex_formulas.norm.lst',
    'data_filter': 'im2latex_train_filter.lst',
    'transform': 'train'
}

val_dataset = {
    'root': 'dataset',
    'images_folder': 'formula_images_processed',
    'label_file': 'im2latex_formulas.norm.lst',
    'data_filter': 'im2latex_validate_filter.lst',
    'transform': 'train'
}

train_dataloader = {
    'batch_size': 128,
    'num_workers': 4
}

val_dataloader = {
    'batch_size': 128,
    'num_workers': 4
}



model = {
    'd_model': 128,
    'dim_feedforward': 256,
    'num_heads': 4,
    'dropout': 0.3,
    'num_decoder_layers': 3,
    'num_classes': 538,
    'max_len': 150
}

optimizer = {
    'lr': 0.001,
    'weight_decay': 0.0001,
    'milestones': [10],
    'gamma': 0.5
}


training = {
    'epochs': 20
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