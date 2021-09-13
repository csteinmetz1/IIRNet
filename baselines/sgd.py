import time
import torch

from iirnet.loss import LogMagTargetFrequencyLoss


class SGDFilterDesign(torch.nn.Module):
    """Design a filter by performing SGD."""

    def __init__(
        self,
        n_iters=1000,
        lr=2e-5,
        schedule_lr=False,
        pole_zero=True,
        verbose=True,
        order=16,
    ):
        super(SGDFilterDesign, self).__init__()
        self.n_iters = n_iters
        self.lr = lr
        self.schedule_lr = schedule_lr
        self.pole_zero = pole_zero
        self.verbose = verbose
        self.order = order

        self.magtarget = LogMagTargetFrequencyLoss()

    def init_sos(self):
        # create the biquad poles and zeros we will optimize
        self.sos = torch.nn.Parameter(torch.ones(1, self.order, 6, requires_grad=True))
        # with torch.no_grad():
        #    self.sos.data.uniform_(0.5, 0.9)

        # setup optimization
        self.optimizer = torch.optim.SGD([self.sos], lr=self.lr)

    def __call__(self, target_dB):

        with torch.enable_grad():
            self.init_sos()
            target_dB = target_dB.to(self.sos.device)
            for n in range(self.n_iters):
                if self.pole_zero:
                    g = self.sos[:, :, 0] + 1
                    poles_real = self.sos[:, :, 1]
                    poles_imag = self.sos[:, :, 2]
                    zeros_real = self.sos[:, :, 3]
                    zeros_imag = self.sos[:, :, 4]

                    # ensure stability
                    pole = torch.complex(poles_real, poles_imag)
                    pole = pole * torch.tanh(pole.abs()) / pole.abs()

                    # ensure zeros inside unit circle
                    zero = torch.complex(zeros_real, zeros_imag)
                    zero = zero * torch.tanh(zero.abs()) / zero.abs()

                    # Apply gain g to numerator by multiplying each coefficient by g
                    b0 = g
                    b1 = g * -2 * zero.real
                    b2 = g * ((zero.real ** 2) + (zero.imag ** 2))
                    a0 = torch.ones(g.shape, device=g.device)
                    a1 = -2 * pole.real
                    a2 = (pole.real ** 2) + (pole.imag ** 2)

                    # reconstruct SOS
                    out_sos = torch.stack([b0, b1, b2, a0, a1, a2], dim=-1)
                else:
                    # extract coefficients
                    b0 = self.sos[:, :, 0]
                    b1 = self.sos[:, :, 1]
                    b2 = self.sos[:, :, 2]
                    a0 = torch.ones(b0.shape, device=b0.device)
                    a1 = self.sos[:, :, 4]
                    a2 = self.sos[:, :, 5]

                    # Eq. 4 from Nercessian et al. 2021
                    a1 = 2 * torch.tanh(a1)

                    # Eq. 5 from above
                    a2 = ((2 - torch.abs(a1)) * torch.tanh(a2) + torch.abs(a1)) / 2

                    # reconstruct SOS
                    out_sos = torch.stack([b0, b1, b2, a0, a1, a2], dim=-1)

                loss = self.magtarget(out_sos, target_dB.view(1, -1))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.schedule_lr:
                    self.scheduler.step()

                if self.verbose:
                    print(f" {n+1} {loss.item():0.2f} dB")

        return out_sos
