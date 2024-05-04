import torch

from torchmetrics import Metric
from torchmetrics.text import BLEUScore, EditDistance, CharErrorRate

# write wrapper for BLEUScore to have it work on batched inputs
class BLEUScoreMetric(Metric):
    def __init__(self, tokenizer, n_gram: int = 4, smooth: bool = False, weights: list = None):
        super().__init__()
        
        self.add_state('score', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        
        self.bleu = BLEUScore(n_gram=n_gram, smooth=smooth, weights=weights)
        self.tokenizer = tokenizer

    def update(self, preds, targets):
        n_batches = preds.size(0)
        for i in range(n_batches):
            pred = preds[i]
            target = targets[i]

            pred = self.tokenizer.decode_to_string(pred)
            target = self.tokenizer.decode_to_string(target)

            self.score += self.bleu([pred], [[target]])
            self.count += 1

    def compute(self):
        return self.score / self.count


# write wrapper for EditDistance to have it work on batched inputs
class EditDistanceMetric(Metric):
    def __init__(self, tokenizer, substitution_cost=1, reduction='mean'):
        super().__init__()
        
        self.add_state('score', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        
        self.edit_distance = EditDistance(substitution_cost=substitution_cost, reduction=reduction)
        self.tokenizer = tokenizer

    def update(self, preds, targets):
        n_batches = preds.size(0)
        for i in range(n_batches):
            pred = preds[i]
            target = targets[i]

            pred = self.tokenizer.decode_to_string(pred)
            target = self.tokenizer.decode_to_string(target)

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
        n_batches = preds.size(0)
        for i in range(n_batches):
            pred = preds[i]
            target = targets[i]

            pred = self.tokenizer.decode_to_string(pred)
            target = self.tokenizer.decode_to_string(target)

            self.score += self.cer(pred, target)
            self.count += 1

    def compute(self):
        return self.score / self.count

