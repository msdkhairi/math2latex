import torch

from torchmetrics import Metric
from torchmetrics.text import BLEUScore, EditDistance, CharErrorRate


class BLEUScoreMetric(Metric):
    def __init__(self, tokenizer, n_gram: int = 4, smooth: bool = False, weights: list = None):
        super().__init__()
        
        self.add_state('score', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        
        self.bleu = BLEUScore(n_gram=n_gram, smooth=smooth, weights=weights)
        self.tokenizer = tokenizer

    def update(self, preds, targets):

        pred = self.tokenizer.decode_to_string(preds[0])
        target = self.tokenizer.decode_to_string(targets[0])

        self.score += self.bleu([pred], [target])
        self.count += 1

    def compute(self):
        return self.score / self.count


class EditDistanceMetric(Metric):
    def __init__(self, tokenizer, substitution_cost=1, reduction='mean'):
        super().__init__()
        
        self.add_state('score', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        
        self.edit_distance = EditDistance(substitution_cost=substitution_cost, reduction=reduction)
        self.tokenizer = tokenizer

    def update(self, preds, targets):

        pred = self.tokenizer.decode_to_string(preds[0])
        target = self.tokenizer.decode_to_string(targets[0])

        self.score += self.edit_distance(pred, target)
        self.count += 1

    def compute(self):
        return self.score / self.count


class CERMetric(Metric):
    def __init__(self, tokenizer):
        super().__init__()
        
        self.add_state('score', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        
        self.cer = CharErrorRate()
        self.tokenizer = tokenizer

    def update(self, preds, targets):

        pred = self.tokenizer.decode_to_string(preds[0])
        target = self.tokenizer.decode_to_string(targets[0])

        self.score += self.cer(pred, target)
        self.count += 1

    def compute(self):
        return self.score / self.count

