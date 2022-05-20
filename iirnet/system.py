import torch
import random
import pytorch_lightning as pl
from argparse import ArgumentParser

from iirnet.mlp import MLPModel
import iirnet.loss as loss
import iirnet.plotting as plotting
import iirnet.signal as signal
import scipy.signal
import numpy as np


class System(pl.LightningModule):
    """Base system module."""

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = MLPModel(
            self.hparams.num_points,
            self.hparams.num_layers,
            self.hparams.hidden_dim,
            self.hparams.model_order,
            self.hparams.normalization,
            self.hparams.output,
        )
        self.magfreqzloss = loss.FreqDomainLoss()
        self.dbmagfreqzloss = loss.LogMagFrequencyLoss()

    def forward(self, x: torch.Tensor):
        return self.model(x)

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

        if zpk is None:
            zs = []
            ps = []
            ks = []
            for n in range(sos.shape[0]):
                z, p, k = scipy.signal.sos2zpk(sos[n, ...].cpu().numpy())
                z = zs.append(
                    torch.complex(
                        torch.tensor(np.real(z)),
                        torch.tensor(np.imag(z)),
                    )
                )
                p = ps.append(
                    torch.complex(
                        torch.tensor(np.real(p)),
                        torch.tensor(np.imag(p)),
                    )
                )
                k = ks.append(torch.tensor(k))
            zs = torch.stack(zs)
            ps = torch.stack(ps)
            ks = torch.stack(ks)

        # move tensors to cpu for logging
        outputs = {
            "pred_sos": pred_sos.cpu(),
            "sos": sos.cpu(),
            "mag_dB": mag_dB.cpu(),
            "z": zs,
            "p": ps,
            "k": ks,
        }

        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        ms1 = int(self.hparams.max_epochs * 0.8)
        ms2 = int(self.hparams.max_epochs * 0.95)
        milestones = [ms1, ms2]
        print(
            "Learning rate schedule:",
            f"1:{self.hparams.lr:0.2e} ->",
            f"{ms1}:{self.hparams.lr*0.1:0.2e} ->",
            f"{ms2}:{self.hparams.lr*0.01:0.2e}",
        )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones,
            gamma=0.1,
            verbose=False,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss",
        }

    # add any model hyperparameters here
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # --- model related ---
        parser.add_argument("--num_points", type=int, default=512)
        parser.add_argument("--num_layers", type=int, default=2)
        parser.add_argument("--hidden_dim", type=int, default=512)
        parser.add_argument("--model_order", type=int, default=10)
        parser.add_argument("--normalization", type=str, default="none")
        parser.add_argument("--output", type=str, default="sos")
        # --- training related ---
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--eps", type=float, default=1e-8)
        parser.add_argument("--priority_order", action="store_true")
        parser.add_argument("--experiment_name", type=str, default="experiment")

        return parser
