import json

from collections import Counter
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab


class Tokenizer:
    def __init__(self, formulas=None, max_len=150):
        # self.tokenizer = get_tokenizer(None)
        self.tokenizer = get_tokenizer("basic_english")
        self.max_len = max_len
        
        if formulas is not None:
            self.vocab = self._build_vocab(formulas)
            self.vocab.set_default_index(self.vocab['<unk>'])
            self.pad_index = self.vocab['<pad>']
            self.ignore_indices = {self.vocab['<pad>'], self.vocab['<bos>'], self.vocab['<eos>'], self.vocab['<unk>']}
        else:
            self.vocab = None

    def _build_vocab(self, formulas):
        counter = Counter()
        for formula in formulas:
            counter.update(self.tokenizer(formula))
        return vocab(counter, specials=['<pad>', '<bos>', '<eos>', '<unk>'], min_freq=2)
    
    def encode(self, formula, with_padding=False):
        tokens = self.tokenizer(formula)
        tokens = ['<bos>'] + tokens + ['<eos>']
        if with_padding:
            tokens = self.pad(tokens, self.max_len)
        # add the bos and eos to begining and end of the tokens
        return [self.vocab[token] for token in tokens]
    
    def decode(self, indices):
        return self.vocab.lookup_tokens(list(indices))
    
    def decode_clean(self, indices):
        # removes the ignore indices from the decoded tokens
        cleaned_indices = [index for index in indices if int(index) not in self.ignore_indices]
        # if self.vocab['<eos>'] in cleaned_indices:
        #     cleaned_indices = cleaned_indices[:cleaned_indices.index(self.vocab['<eos>'])]
        return self.vocab.lookup_tokens(cleaned_indices)
    
    def decode_to_string(self, tokens):
        # returns the decoded tokens as a string
        decoded = self.decode_clean(tokens)
        return ' '.join(decoded)


    def pad(self, tokens, max_len):
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
            tokens[-1] = '<eos>'
            return tokens
        return tokens + ['<pad>'] * (max_len - len(tokens))

    def save_vocab(self, file_path="dataset/tokenizer_vocab.json"):
        # Save the list of tokens which reflects both `itos` and `stoi`
        vocab_data = {
            'itos': self.vocab.get_itos()
        }
        with open(file_path, 'w') as f:
            json.dump(vocab_data, f)

    def load_vocab(self, file_path):
        with open(file_path, 'r') as f:
            vocab_data = json.load(f)
        # Reconstruct the vocabulary from the itos list
        ordered_tokens = vocab_data['itos']
        # Reconstruct the counter from the ordered list
        counter = Counter({token: idx + 1 for idx, token in enumerate(ordered_tokens)})  # idx+1 to ensure non-zero freq
        self.vocab = vocab(counter, specials=['<pad>', '<bos>', '<eos>', '<unk>'])
        self.vocab.set_default_index(self.vocab['<unk>'])
        self.pad_index = self.vocab['<pad>']
        self.ignore_indices = {self.vocab['<pad>'], self.vocab['<bos>'], self.vocab['<eos>'], self.vocab['<unk>']}


    def __len__(self):
        return len(self.vocab)