import torch
import torch.nn as nn

import torchvision

from .positional_encoding import PositionalEncoding1D, PositionalEncoding2D 

class ResNetTransformer(nn.Module):
    def __init__(self,
                 d_model, 
                 num_heads, 
                 num_encoder_layers, 
                 num_decoder_layers, 
                 dim_feedforward, 
                 dropout=0.1, 
                 pos_enc_dropout=0.1, 
                 activation='relu',
                 max_len=150, 
                 num_classes=1000):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.num_classes = num_classes

        # Encoder
        resnet18 = torchvision.models.resnet18(pretrained=True)
        # Remove the classification head from resnt18
        self.backbone = nn.Sequential(*list(resnet18.children())[:-2])

        self.conv1x1 = nn.Conv2d(512, d_model, kernel_size=1, stride=1, padding=0)
        self.encoder_pos_enc = PositionalEncoding2D(d_model)

        # Decoder
        self.embedding = nn.Embedding(num_classes, d_model)
        self.decoder_pos_enc = PositionalEncoding1D(d_model, dropout=pos_enc_dropout, max_len=max_len)
        
        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, dim_feedforward, dropout, activation)
        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_decoder_layers)

        self.fc = nn.Linear(d_model, num_classes)
        self.softmax = nn.Softmax(dim=2)


    def encode(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        print("x.shape: ", x.shape)
        x = self.backbone(x)
        print("x.shape: ", x.shape)
        x = self.conv1x1(x)
        print("x.shape: ", x.shape)
        x = x.permute(0, 2, 3, 1)
        print("x.shape: ", x.shape)
        x = x.view(x.size(0), -1, x.size(-1))
        print("x.shape: ", x.shape)
        x = self.encoder_pos_enc(x)
        return x
    
    def decode(self, tgt, x):
        tgt = self.embedding(tgt)
        tgt = self.decoder_pos_enc(tgt)
        tgt = tgt.permute(1, 0, 2)

        output = self.transformer_decoder(tgt, x)
        output = self.fc(output)
        return output

    def forward(self, x, tgt):
        # Encoder
        x = self.encode(x)

        # Decoder
        output = self.decode(tgt, x)
        # output = self.softmax(output)
        return output
    
    
    


