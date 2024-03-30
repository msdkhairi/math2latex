dataset = {
    'root': 'dataset',
    'images_folder': 'formula_images_processed',
    'label_file': 'im2latex_formulas.norm.lst',
    'data_filter': 'im2latex_train_filter.lst',
    'transform': 'train',
    'batch_size': 8,
    'num_workers': 4
}

train_dataloader = {
    'batch_size': 2,
    'num_workers': 1
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
    'lr': 0.001
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
    'dataset': Config(dataset),
    'train_dataloader': Config(train_dataloader),
    'model': Config(model),
    'optimizer': Config(optimizer),
    'training': Config(training)
})