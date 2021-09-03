import torch
import torch.fft
import numpy as np

import iirnet.signal as signal


class LogMagFrequencyLoss(torch.nn.Module):
    def __init__(self, priority=False):
        super(LogMagFrequencyLoss, self).__init__()
        self.priority = priority

    def forward(self, input, target, eps=1e-8):
        bs = input.size(0)
        loss = 0

        if False:
            for n in range(bs):
                w, input_h = signal.sosfreqz(input[n, ...])
                w, target_h = signal.sosfreqz(target[n, ...])

                input_mag = 20 * torch.log10(signal.mag(input_h) + eps)
                target_mag = 20 * torch.log10(signal.mag(target_h) + eps)

                loss += torch.nn.functional.l1_loss(input_mag, target_mag)
        elif self.priority:
            # in this case, we compute the loss comparing the response as we increase the number
            # of biquads in the cascade, this should encourage to use lower order filter.

            # first compute the target response
            w, target_h = signal.sosfreqz(target, log=False)
            target_mag = 20 * torch.log10(signal.mag(target_h) + eps)

            n_sections = input.shape[1]
            mag_loss = 0
            # now compute error with each group of biquads
            for n in np.arange(n_sections, step=2):

                sos = input[:, 0 : n + 2, :]
                w, input_h = signal.sosfreqz(sos, log=False)
                input_mag = 20 * torch.log10(signal.mag(input_h) + eps)

                mag_loss += torch.nn.functional.mse_loss(input_mag, target_mag)

        else:
            w, input_h = signal.sosfreqz(input, log=False)
            w, target_h = signal.sosfreqz(target, log=False)

            input_mag = 20 * torch.log10(signal.mag(input_h) + eps)
            target_mag = 20 * torch.log10(signal.mag(target_h) + eps)

            mag_loss = torch.nn.functional.mse_loss(input_mag, target_mag)

        return mag_loss


class FreqDomainLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, eps=1e-8):
        bs = input.size(0)
        loss = 0

        w, input_h = signal.sosfreqz(input, log=False)
        w, target_h = signal.sosfreqz(target, log=False)

        input_mag = signal.mag(input_h)
        target_mag = signal.mag(target_h)

        input_mag_log = torch.log(input_mag)
        target_mag_log = torch.log(target_mag)

        mag_log_loss = torch.nn.functional.l1_loss(input_mag_log, target_mag_log)

        return mag_log_loss

    @staticmethod
    def compute_dB_magnitude(mag, eps=1e-16):
        mag_dB = 20 * torch.log10(mag + eps)
        mag_dB = torch.clamp(mag_dB, -128, 128)
        mag_dB /= 128.0  # scale between -1 and 1
        return mag_dB


class LogMagTargetFrequencyLoss(torch.nn.Module):
    def __init__(self, priority=False, use_dB=False, zero_mean=True):
        super(LogMagTargetFrequencyLoss, self).__init__()
        self.priority = priority
        self.use_dB = use_dB
        self.zero_mean = zero_mean

    def forward(self, input_sos, target_mag, eps=1e-8):

        if self.zero_mean:
            target_mag = target_mag - target_mag.mean(dim=-1, keepdim=True)

        if self.priority:
            n_sections = input_sos.shape[1]
            mag_loss = 0
            # now compute error with each group of biquads
            for n in np.arange(n_sections, step=2):
                sos = input_sos[:, 0 : n + 2, :]
                w, input_h = signal.sosfreqz(sos, worN=target_mag.shape[-1], log=False)
                input_mag = 20 * torch.log10(signal.mag(input_h) + eps).float()
                mag_loss += torch.nn.functional.mse_loss(input_mag, target_mag)
        else:
            w, input_h = signal.sosfreqz(
                input_sos, worN=target_mag.shape[-1], log=False
            )
            if self.use_dB:
                input_mag = 20 * torch.log10(signal.mag(input_h) + eps).float()
            else:
                input_mag_log = torch.log(signal.mag(input_h) + eps).float()
                input_mag_lin = signal.mag(input_h).float()

            mag_loss = torch.nn.functional.mse_loss(input_mag, target_mag)

        return mag_loss


class ComplexLoss(torch.nn.Module):
    def __init__(self, threshold=1e-16):
        super(ComplexLoss, self).__init__()
        self.threshold = threshold

    def forward(self, input, target):
        bs = input.size(0)
        loss = 0

        if False:
            for n in range(bs):

                input_sos = input[n, ...]
                target_sos = target[n, ...]

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
        out_sos = sos[sos.sum(-1) > self.threshold, :]

        # check if all sections were removed
        if out_sos.size(0) == 0:
            out_sos = sos

        return out_sos
