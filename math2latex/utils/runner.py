import os
import torch

import lightning as L
from data import Tokenizer, MathToLatexDataset
from model import ResNetTransformer
from .metrics import BLEUScoreMetric, EditDistanceMetric, CERMetric

class LitMathToLatex(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.config = config

        self.tokenizer = Tokenizer()
        self.tokenizer.load_vocab("tokenizer_vocab.json")

        self.model = ResNetTransformer(
            d_model=self.config.model.d_model,
            num_heads=self.config.model.num_heads,
            num_decoder_layers=self.config.model.num_decoder_layers,
            dim_feedforward=self.config.model.dim_feedforward,
            dropout=self.config.model.dropout,
            max_len_output=self.config.model.max_len,
            num_classes=self.config.model.num_classes
        )

        self.loss = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_index)

        self.val_bleu = BLEUScoreMetric(self.tokenizer)
        self.val_edit_distance = EditDistanceMetric(self.tokenizer)
        self.val_cer = CERMetric(self.tokenizer)
        
    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images, targets[:, :-1])
        loss = self.loss(outputs, targets[:, 1:])
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images, targets[:, :-1])
        loss = self.loss(outputs, targets[:, 1:])
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        preds = self.model.predict(images)

        val_bleu = self.val_bleu(preds, targets)
        val_edit_distance = self.val_edit_distance(preds, targets)
        val_cer = self.val_cer(preds, targets)

        self.log('val_bleu', val_bleu, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_ed', val_edit_distance, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_cer', val_cer, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model.predict(images)
        test_bleu = self.val_bleu(outputs, targets)
        test_edit_distance = self.val_edit_distance(outputs, targets)
        test_cer = self.val_cer(outputs, targets)
        self.log('test_bleu', test_bleu)
        self.log('test_ed', test_edit_distance)
        self.log('test_cer', test_cer)
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=self.config.optimizer.milestones, 
            gamma=self.config.optimizer.gamma)
        
        return [optimizer], [scheduler]





class Runner:
    def __init__(self, config):
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.ignore_index)

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
        return loss.item()
    
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
            pbar = tqdm(self.train_dataloader, desc=f'Epoch {epoch}')
            for i, batch in enumerate(pbar):
                loss = self.train_step(batch)
                epoch_loss += loss
                iter_count += 1
                pbar.set_postfix({'loss': loss})
                if i % 100 == 0 and epoch_loss:
                    writer.add_scalar('Train Loss', loss, epoch * len(self.train_dataloader) + i)
                    writer.add_scalar('Learning Rate', self.optimizer.param_groups[0]['lr'], epoch * len(self.train_dataloader) + i)
            writer.add_scalar('Train Loss Epoch', epoch_loss/iter_count, epoch)
            pbar.close()

            val_loss = 0
            val_iter_count = 0
            for i, batch in enumerate(self.val_dataloader):
                loss = self.val_step(batch)
                val_loss += loss
                val_iter_count += 1
            logger.info(f'Epoch: {epoch} - Val Loss: {val_loss/val_iter_count}')
            writer.add_scalar('Val Loss', val_loss/val_iter_count, epoch)

            self.lr_scheduler.step()

            # check if runs directory exists otherwise create it
            if not os.path.exists('runs'):
                os.makedirs('runs')
            self.save_model(f'runs/model_epoch_{epoch+1}.pth')

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
            
                    
    

                
        

