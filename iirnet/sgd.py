import time
import torch

from iirnet.loss import LogMagTargetFrequencyLoss

class SGDFilterDesign(torch.nn.Module):
    
    def __init__(self, n_iters=1000, lr=1e-1):
        super(SGDFilterDesign, self).__init__()
        self.n_iters = n_iters
        self.lr = lr

        # create the biquad poles and zeros we will optimize
        self.sos = torch.rand(16,6, requires_grad=True)

        self.optimizer = torch.optim.SGD([self.sos], lr=self.lr)
        self.magtarget = LogMagTargetFrequencyLoss()

    def __call__(self, target_dB):

        with torch.enable_grad():
            for n in range(self.n_iters):
                loss = self.magtarget(self.sos, target_dB.view(1,-1))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return self.sos