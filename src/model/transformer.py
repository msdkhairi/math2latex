import torch
import torch.nn as nn

import torchvision

from .positional_encoding import PositionalEncoding1D, PositionalEncoding2D 

class ResNetTransformer(nn.Module):
    def __init__(self,
                 d_model=128, 
                 num_heads=4,
                 num_decoder_layers=3, 
                 dim_feedforward=256, 
                 dropout=0.1, 
                 pos_enc_dropout=0.1, 
                 activation='relu',
                 max_len_output=150, 
                 num_classes=462):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.num_classes = num_classes
        self.max_len_output = max_len_output

        # Encoder
        resnet18 = torchvision.models.resnet18(weights=None)

        # Remove the classification head and layer 4 from resnt18 and keep the first 3 layers
        self.backbone = nn.Sequential(*list(resnet18.children())[:-3])

        self.conv1x1 = nn.Conv2d(256, d_model, kernel_size=1, stride=1, padding=0)
        self.encoder_pos_enc = PositionalEncoding2D(d_model, 
                                                    max_h=1000, 
                                                    max_w=1000, 
                                                    dropout=pos_enc_dropout) # no images are larger than 1000x1000

        # Decoder
        self.embedding = nn.Embedding(num_classes, d_model)
        self.decoder_pos_enc = PositionalEncoding1D(d_model, 
                                                    max_len=max_len_output, 
                                                    dropout=pos_enc_dropout)
        _transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, 
                                                                nhead=num_heads, 
                                                                dim_feedforward=dim_feedforward, 
                                                                dropout=dropout, 
                                                                activation=activation)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer=_transformer_decoder_layer,
                                                         num_layers=num_decoder_layers)

        self.linear = nn.Linear(d_model, num_classes)
        # self.softmax = nn.Softmax(dim=2)

        # get target mask for training
        self.tgt_mask = self.get_tgt_mask(max_len_output)
        if self.training:
            self._init_weights()


    def _init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.zero_()

        nn.init.kaiming_normal_(self.conv1x1.weight, mode='fan_out', nonlinearity='relu')
        if self.conv1x1.bias is not None:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.conv1x1.weight)
            bound = 1 / torch.sqrt(torch.tensor(fan_out))
            nn.init.normal_(self.conv1x1.bias, -bound, bound)

    def get_tgt_mask(self, target_size):
        tgt_mask = torch.triu(torch.ones(target_size, target_size), diagonal=1)
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float('-inf'))
        return tgt_mask

    def encode(self, x):
        # Repeat the input x if it has only 1 channel
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.backbone(x)
        x = self.conv1x1(x)
        x = self.encoder_pos_enc(x)
        x = x.flatten(2)
        x = x.permute(2, 0, 1)
        return x
    
    def decode(self, tgt, x):
        tgt = tgt.permute(1, 0)
        tgt = self.embedding(tgt)
        tgt = self.decoder_pos_enc(tgt)
        tgt_mask = self.tgt_mask[:tgt.size(0), :tgt.size(0)]
        output = self.transformer_decoder(tgt, x, tgt_mask)
        output = self.linear(output)
        return output

    def forward(self, x, tgt):
        # Encoder
        x = self.encode(x)

        # Decoder
        output = self.decode(tgt, x)
        return output.permute(1, 2, 0)
    

    def predict(self, x):
        x = self.encode(x)
        tgt = torch.zeros(self.max_len_output_, x.size(1)).long().to(x.device)
        for i in range(self.max_len_output):
            output = self.decode(tgt[:i+1], x)
            output = self.fc(output)
            output = output[-1].argmax()
            tgt[i] = output
        return tgt.permute(1, 0)    
    
