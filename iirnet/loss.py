import torch
import torch.fft
import numpy as np

import iirnet.signal as signal

class LogMagFrequencyLoss(torch.nn.Module):
    def __init__(self):
        super(LogMagFrequencyLoss, self).__init__()

    def forward(self, input, target):
        bs = input.size(0)
        loss = 0
        for n in range(bs):
            w, input_h = signal.sosfreqz(input[n,...])
            w, target_h = signal.sosfreqz(target[n,...])

            input_mag = torch.log(signal.mag(input_h))
            target_mag = torch.log(signal.mag(target_h))

            loss += torch.nn.functional.l1_loss(input_mag, target_mag)
        
        return loss / bs

class ComplexLoss(torch.nn.Module):
    def __init__(self, threshold=1e-16):
        super(ComplexLoss, self).__init__()
        self.threshold = threshold

    def forward(self, input, target):
        bs = input.size(0)
        loss = 0

        if False:
            for n in range(bs):

                input_sos = input[n,...]
                target_sos = target[n,...]

                if self.threshold is not None:
                    input_sos = self.apply_threshold(input_sos)
                    target_sos = self.apply_threshold(target_sos)

                w, input_h = signal.sosfreqz(input_sos, log=True)
                w, target_h = signal.sosfreqz(target_sos, log=True)

                real_loss = torch.nn.functional.l1_loss(input_h.real, target_h.real)
                imag_loss = torch.nn.functional.l1_loss(input_h.imag, target_h.imag)
                loss += real_loss + imag_loss
        else:
            w, input_h = signal.sosfreqz(input, log=False)
            w, target_h = signal.sosfreqz(target, log=False)
            real_loss = torch.nn.functional.mse_loss(input_h.real, target_h.real)
            imag_loss = torch.nn.functional.mse_loss(input_h.imag, target_h.imag)
            loss = real_loss + imag_loss

        return torch.mean(loss)
        
    def apply_threshold(self, sos):
        out_sos = sos[sos.sum(-1) > self.threshold,:]

        # check if all sections were removed
        if out_sos.size(0) == 0:
            out_sos = sos

        return out_sos