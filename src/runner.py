import torch

from .data.dataset import TrainDataset, get_dataloader
from .model.transformer2 import ResNetTransformer

class Runner:
    def __init__(self, config):
        self.config = config

        self.train_dataset = TrainDataset(
            self.config.dataset.root,
            self.config.dataset.images_folder,
            self.config.dataset.label_file,
            self.config.dataset.data_filter,
            transform=self.config.dataset.transform,
        )
        
        self.train_dataloader = get_dataloader(
            self.train_dataset,
            self.config.train_dataloader.batch_size,
            self.config.train_dataloader.num_workers,
        )

        self.model = ResNetTransformer(
            d_model=self.config.model.d_model,
            dim_feedforward=self.config.model.dim_feedforward,
            num_heads=self.config.model.num_heads,
            dropout=self.config.model.dropout,
            num_decoder_layers=self.config.model.num_decoder_layers,
            num_classes=self.config.model.num_classes,
            max_output_len=self.config.model.max_len,
            sos_index=1,
            eos_index=2,
            pad_index=0
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.optimizer.lr)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=self.model.pad_index)

    def train_step(self, batch):
        self.optimizer.zero_grad()
        images, targets = batch
        output = self.model(images, targets)
        loss = self.loss(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def train(self):
        for epoch in range(self.config.training.epochs):
            epoch_loss = []
            for iter, batch in enumerate(self.train_dataloader):
                loss = self.train_step(batch)
                epoch_loss.append(loss.item())
                if iter % 100 == 0 and epoch_loss:
                    print('Epoch:', epoch, 'Iteration:', iter, 
                          'Loss:', torch.mean(torch.tensor(epoch_loss)))

                
        

