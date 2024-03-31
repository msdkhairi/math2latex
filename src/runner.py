import torch

from data.dataset import TrainDataset, get_dataloader
from model.transformer2 import ResNetTransformer
import logging
from torch.utils.tensorboard import SummaryWriter

class Runner:
    def __init__(self, config):
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_dataset = TrainDataset(
            self.config.train_dataset.root,
            self.config.train_dataset.images_folder,
            self.config.train_dataset.label_file,
            self.config.train_dataset.data_filter,
            transform=self.config.train_dataset.transform,
        )
        
        self.train_dataloader = get_dataloader(
            self.train_dataset,
            self.config.train_dataloader.batch_size,
            self.config.train_dataloader.num_workers,
        )

        self.val_dataset = TrainDataset(
            self.config.val_dataset.root,
            self.config.val_dataset.images_folder,
            self.config.val_dataset.label_file,
            self.config.val_dataset.data_filter,
            transform=self.config.val_dataset.transform,
        )

        self.val_dataloader = get_dataloader(
            self.val_dataset,
            self.config.val_dataloader.batch_size,
            self.config.val_dataloader.num_workers,
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
        ).to(self.device)

        self.loss = torch.nn.CrossEntropyLoss(ignore_index=self.model.pad_index)

        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                        lr=self.config.optimizer.lr, 
                                        weight_decay=self.config.optimizer.weight_decay)
        
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, 
            milestones=self.config.optimizer.milestones, 
            gamma=self.config.optimizer.gamma)

    def train_step(self, batch):
        self.optimizer.zero_grad()
        images, targets = batch
        images = images.to(self.device)
        targets = targets.to(self.device)
        output = self.model(images, targets[:, :-1])
        loss = self.loss(output, targets[:, 1:])
        loss.backward()
        self.optimizer.step()
        return loss
    
    def train(self):
        # Create a logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler('training.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Create a TensorBoard writer
        writer = SummaryWriter()

        for epoch in range(self.config.training.epochs):
            epoch_loss = 0
            iter_count = 0
            for i, batch in enumerate(self.train_dataloader):
                loss = self.train_step(batch)
                loss = loss.item()
                epoch_loss += loss
                iter_count += 1
                if i % 100 == 0 and epoch_loss:
                    logger.info(f'Epoch: {epoch} - Iteration: {i} - Loss: {loss}')
                    writer.add_scalar('Train Loss', loss, epoch * len(self.train_dataloader) + i)
                    writer.add_scalar('Learning Rate', self.optimizer.param_groups[0]['lr'], epoch * len(self.train_dataloader) + i)
            writer.add_scalar('Train Loss Epoch', epoch_loss/iter_count, epoch)
            logger.info(f'Epoch: {epoch} - Train Loss: {epoch_loss/iter_count}')
            
            val_loss = 0
            val_iter_count = 0
            for i, batch in enumerate(self.val_dataloader):
                loss = self.val_step(batch)
                val_loss += loss
                val_iter_count += 1
            logger.info(f'Epoch: {epoch} - Val Loss: {val_loss/val_iter_count}')
            writer.add_scalar('Val Loss', val_loss/val_iter_count, epoch)

            self.lr_scheduler.step()

            self.save_model(f'model_epoch_{epoch}.pth')

        # Close the TensorBoard writer
        writer.close()
            
                    
    def val_step(self, batch):
        with torch.no_grad():
            images, targets = batch
            images = images.to(self.device)
            targets = targets.to(self.device)
            output = self.model(images, targets[:, :-1])
            loss = self.loss(output, targets[:, 1:])
            return loss.item()
                    
    def predict(self, image):
        self.model.eval()
        with torch.no_grad():
            image = image.to(self.device)
            output = self.model(image)
            return output

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
            
                    
    

                
        

