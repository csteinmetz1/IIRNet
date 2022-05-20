import torch
import pytorch_lightning as pl
from argparse import ArgumentParser

from .system import IIRNet

class LSTMModel(IIRNet):
    """ LSTM module. """
    def __init__(self, 
                num_points = 512,
                num_layers = 1,
                hidden_dim = 32,
                max_order = 10,
                lr = 1e-4,
                **kwargs):
        super(LSTMModel, self).__init__()

        self.save_hyperparameters()

        self.lstm = torch.nn.LSTM(self.hparams.num_points,
                                self.hparams.hidden_dim,
                                self.hparams.num_layers)
        self.output = torch.nn.Linear(self.hparams.hidden_dim, 6)

    def forward(self, mag, phs=None):
        
        mag = mag.unsqueeze(-1) # add sequence dim
        # create a sequence of max length 
        seq = mag.permute(2,0,1)
        seq = seq.repeat(self.hparams.max_order//2,1,1)

        out, _ = self.lstm(seq)
        out = self.output(out)
        sos = out.permute(1,0,2)

        return sos

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    # add any model hyperparameters here
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # --- model related ---
        parser.add_argument('--num_points', type=int, default=512)
        parser.add_argument('--num_layers', type=int, default=4)
        parser.add_argument('--hidden_dim', type=int, default=32)
        parser.add_argument('--max_order', type=int, default=10)
        # --- training related ---
        parser.add_argument('--lr', type=float, default=1e-3)

        return parser
