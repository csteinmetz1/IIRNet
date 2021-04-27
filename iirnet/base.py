import torch
import random
import pytorch_lightning as pl
from argparse import ArgumentParser

import iirnet.loss as loss
import iirnet.plotting as plotting
import iirnet.signal as signal

class IIRNet(pl.LightningModule):
    """ Base IIRNet module. """
    def __init__(self, **kwargs):
        super(IIRNet, self).__init__()
        self.magfreqzloss = loss.LogMagTargetFrequencyLoss(priority=False)
        self.magfreqzloss_val = loss.LogMagTargetFrequencyLoss(priority=False)

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        mag, phs, real, imag, sos = batch
        pred_sos = self(mag)
        loss = self.magfreqzloss(pred_sos, mag)

        self.log('train_loss', 
                    loss, 
                    on_step=True, 
                    on_epoch=True, 
                    prog_bar=True, 
                    logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mag, phs, real, imag, sos = batch
        pred_sos = self(mag)
        loss = self.magfreqzloss_val(pred_sos, mag)
        priority_loss = self.magfreqzloss(pred_sos, mag)

        self.log('val_loss', loss)
        self.log('val_loss/priority', priority_loss)

        # move tensors to cpu for logging
        outputs = {
            "pred_sos" : pred_sos.cpu(),
            "sos": sos.cpu(),
            "mag"  : mag.cpu()}

        return outputs

    def validation_epoch_end(self, validation_step_outputs):

        random.shuffle(validation_step_outputs)
        pred_sos = torch.split(validation_step_outputs[0]["pred_sos"], 1, dim=0)       
        sos = torch.split(validation_step_outputs[0]["sos"], 1, dim=0)
        mag = torch.split(validation_step_outputs[0]["mag"], 1, dim=0)

        self.logger.experiment.add_image(
                        "mag-grid", 
                        plotting.plot_response_grid(pred_sos, target_mags=mag), 
                        self.global_step)


    # add any model hyperparameters here
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # --- model related ---
        parser.add_argument('--num_points', type=int, default=512)
        parser.add_argument('--num_layers', type=int, default=4)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--filter_order', type=int, default=2)
        # --- training related ---
        parser.add_argument('--lr', type=float, default=1e-3)

        return parser