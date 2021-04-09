import torch
import pytorch_lightning as pl
from argparse import ArgumentParser

from .base import IIRNet

class MLPModel(IIRNet):
    """ Multi-layer perceptron module. """
    def __init__(self, 
                num_points = 512,
                num_layers = 2,
                hidden_dim = 8192,
                max_order = 2,
                normalization = "none",
                lr = 3e-4,
                **kwargs):
        super(MLPModel, self).__init__()
        self.save_hyperparameters()

        self.layers = torch.nn.ModuleList()

        for n in range(self.hparams.num_layers):
            in_features = self.hparams.hidden_dim if n != 0 else self.hparams.num_points
            out_features = self.hparams.hidden_dim
            if n+1 == self.hparams.num_layers: # no activation at last layer
                my_layer = torch.nn.Linear(in_features, out_features)
                my_layer.bias.data.fill_(0.5) # what is motivation for this init?
                self.layers.append(torch.nn.Sequential(
                    my_layer,
                ))
            else:
                self.layers.append(torch.nn.Sequential(
                    torch.nn.Linear(in_features, out_features),
                    torch.nn.LayerNorm(out_features),
                    torch.nn.LeakyReLU(0.2),
                ))

        n_coef = (self.hparams.max_order//2) * 6
        self.layers.append(torch.nn.Linear(out_features, n_coef))

        if self.hparams.normalization == "bn":
            self.bn = torch.nn.BatchNorm1d(self.hparams.num_points*2)

    def forward(self, mag, phs=None):
        #x = torch.cat((mag), dim=-1)
        x = mag

        if self.hparams.normalization == "tanh":
            x = torch.tanh(x)
        elif self.hparams.normalization == "bn":
            x = self.bn(x)
        elif self.hparams.normalization == "mean":
            x = x - torch.mean(x) # this likely does not do what we want

        for layer in self.layers:
            x = layer(x) 

        # reshape into sos format (n_section, (b0, b1, b2, a0, a1, a2))
        n_sections = self.hparams.max_order//2
        x = x.view(-1,n_sections,6)

        # replace a0
        x[:,:,3] = 1.0

        #x = torch.tanh(x)

        return x

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr)
        #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, verbose=True)
        #lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, verbose=True)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                                                  optimizer, 
                                                  self.hparams.max_epochs, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': 'val_loss'
        }

    # add any model hyperparameters here
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # --- model related ---
        parser.add_argument('--num_points', type=int, default=512)
        parser.add_argument('--num_layers', type=int, default=2)
        parser.add_argument('--hidden_dim', type=int, default=8192)
        parser.add_argument('--max_order', type=int, default=10)
        parser.add_argument('--normalization', type=str, default="none")
        # --- training related ---
        parser.add_argument('--lr', type=float, default=1e-3)

        return parser