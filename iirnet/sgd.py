import time
import torch

from iirnet.loss import LogMagTargetFrequencyLoss

class SGDFilterDesign(torch.nn.Module):
    """ Design a filter by performing SGD.

    Note: I have tried to use the pole-zero representation, 
    but this does not appear to converge. 

    """
    def __init__(self, 
                n_iters=1000, 
                lr=1e-1, 
                schedule_lr=False, 
                pole_zero=False, 
                verbose=False
            ):
        super(SGDFilterDesign, self).__init__()
        self.n_iters = n_iters
        self.lr = lr
        self.schedule_lr = schedule_lr
        self.pole_zero = pole_zero
        self.verbose = verbose
        self.coefs = torch.rand(1,16,6, requires_grad=False)

        self.magtarget = LogMagTargetFrequencyLoss()

    def init_sos(self):
        # create the biquad poles and zeros we will optimize
        self.sos = torch.rand(1,16,6, requires_grad=True)
        with torch.no_grad():
            self.sos.data = self.coefs.data

        # setup optimization
        self.optimizer = torch.optim.SGD([self.sos], lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                                                    self.optimizer, 
                                                    self.n_iters)

    def __call__(self, target_dB):

        with torch.enable_grad():
            self.init_sos()
            for n in range(self.n_iters):
                if self.pole_zero:
                    g = self.sos[:,:,0] 
                    poles_real = self.sos[:,:,1]
                    poles_imag = self.sos[:,:,2]
                    zeros_real = self.sos[:,:,3]
                    zeros_imag = self.sos[:,:,4]

                    # ensure stability
                    pole = torch.complex(poles_real, poles_imag)
                    pole = pole * torch.tanh(pole.abs()) / pole.abs()

                    #ensure zeros inside unit circle
                    zero = torch.complex(zeros_real, zeros_imag)
                    zero = zero * torch.tanh(zero.abs()) / zero.abs()

                    #Apply gain g to numerator by multiplying each coefficient by g 
                    b0 = g
                    b1 = g * -2 * zero.real
                    b2 = g * ((zero.real ** 2) + (zero.imag ** 2))
                    a0 = torch.ones(g.shape, device=g.device)
                    a1 = -2 * pole.real
                    a2 = (pole.real ** 2) + (pole.imag ** 2)

                    # reconstruct SOS
                    out_sos = torch.stack([b0, b1, b2, a0, a1, a2], dim=-1)
                else:
                    out_sos = self.sos

                loss = self.magtarget(out_sos, target_dB.view(1,-1))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.schedule_lr:
                    self.scheduler.step()

                if self.verbose:
                    print(f" {n+1} {loss.item():0.2f} dB")

        return out_sos