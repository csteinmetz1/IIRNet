import torch
import random
import pytorch_lightning as pl
from argparse import ArgumentParser

import iirnet.loss as loss
import iirnet.plotting as plotting
import iirnet.signal as signal


class IIRNet(pl.LightningModule):
    """Base IIRNet module."""

    def __init__(self, **kwargs):
        super(IIRNet, self).__init__()

        self.magfreqzloss = loss.FreqDomainLoss()
        self.dbmagfreqzloss = loss.LogMagFrequencyLoss()

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        mag_dB, mag_dB_norm, phs, real, imag, sos = batch
        pred_sos, _ = self(mag_dB_norm)
        loss = self.magfreqzloss(pred_sos, sos)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        mag_dB, mag_dB_norm, phs, real, imag, sos = batch
        pred_sos, zpk = self(mag_dB_norm)
        loss = self.magfreqzloss(pred_sos, sos)
        dB_MSE = self.dbmagfreqzloss(pred_sos, sos)

        self.log("val_loss", loss, on_step=False)
        self.log("dB_MSE", dB_MSE, on_step=False)

        # move tensors to cpu for logging
        outputs = {
            "pred_sos": pred_sos.cpu(),
            "sos": sos.cpu(),
            "mag_dB": mag_dB.cpu(),
            "z": zpk[0].cpu(),
            "p": zpk[1].cpu(),
            "k": zpk[2].cpu(),
        }

        return outputs

    # add any model hyperparameters here
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # --- model related ---
        parser.add_argument("--num_points", type=int, default=512)
        parser.add_argument("--num_layers", type=int, default=4)
        parser.add_argument("--hidden_dim", type=int, default=128)
        parser.add_argument("--filter_order", type=int, default=2)
        # --- training related ---
        parser.add_argument("--lr", type=float, default=1e-3)

        return parser
