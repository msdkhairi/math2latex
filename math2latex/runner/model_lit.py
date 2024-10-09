import os
import torch

import lightning as L
from torchmetrics import MetricCollection

from data import Tokenizer
from model import ResNetTransformer
from utils import BLEUScoreMetric, EditDistanceMetric, CERMetric

class LitMathToLatex(L.LightningModule):
    def __init__(self,
                # model params
                d_model: int = 128,
                num_heads: int = 4,
                num_decoder_layers: int = 3,
                dim_feedforward: int = 256,
                dropout: float = 0.3,
                num_classes: int = 462,
                max_len: int = 150,
                # optimizer params
                lr: float = 0.001,
                weight_decay: float = 0.0001,
                milestones: list = [10],
                gamma: float = 0.5,
    ):
    # def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        # self.config = config

        self.optim_params = {
            'lr': lr,
            'weight_decay': weight_decay,
            'milestones': milestones,
            'gamma': gamma
        }

        self.tokenizer = Tokenizer()
        self.tokenizer.load_vocab("tokenizer_vocab.json")

        self.model = ResNetTransformer(
            d_model=d_model,
            num_heads=num_heads,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len_output=max_len,
            num_classes=num_classes
        )

        self.loss = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_index)

        metrics = MetricCollection(
            metrics={
            'BLEU': BLEUScoreMetric(self.tokenizer),
            'ED': EditDistanceMetric(self.tokenizer),
            'CER': CERMetric(self.tokenizer),
            },
            compute_groups=True
        )

        # self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        
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
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        preds = self.model.predict(images)

        self.val_metrics.update(preds, targets)

        return loss
    
    def on_validation_epoch_end(self):

        val_results = self.val_metrics.compute()

        # self.logger.log_metrics(metrics=val_results, step=self.global_step)
        self.log('val_bleu', val_results['val_BLEU'].item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_ed', val_results['val_ED'].item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_cer', val_results['val_CER'].item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # reset the metrics
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model.predict(images)
        self.test_metrics.update(outputs, targets)
        return
    
    def on_test_epoch_end(self, outputs):
        test_results = self.test_metrics.compute()
        
        self.log('test_bleu', test_results['test_BLEU'].item())
        self.log('test_ed', test_results['test_ED'].item())
        self.log('test_cer', test_results['test_CER'].item())

        self.test_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.optim_params['lr'],
            weight_decay=self.optim_params['weight_decay'])

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=self.optim_params['milestones'],
            gamma=self.optim_params['gamma'])
        
        return [optimizer], [scheduler]
