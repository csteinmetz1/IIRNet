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
    def __init__(self):
        super(ComplexLoss, self).__init__()

    def forward(self, input, target):
        bs = input.size(0)
        loss = 0
        for n in range(bs):
            w, input_h = signal.sosfreqz(input[n,...])
            w, target_h = signal.sosfreqz(target[n,...])

            real_loss = torch.nn.functional.l1_loss(input_h.real, target_h.real)
            imag_loss = torch.nn.functional.l1_loss(input_h.imag, target_h.imag)
            loss += real_loss + imag_loss
        
        return loss / bs
