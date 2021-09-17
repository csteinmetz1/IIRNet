import numpy as np
from pytorch_lightning.callbacks import Callback

from iirnet.plotting import plot_response_grid


class LogZPKCallback(Callback):
    def __init__(self, num_examples=4):
        super().__init__()
        self.num_examples = 4

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        if outputs is not None and batch_idx == 0:
            examples = np.min([self.num_examples, outputs["z"].shape[0]])
            for n in range(examples):
                z = outputs["z"][batch_idx, ...]
                p = outputs["p"][batch_idx, ...]
                k = outputs["k"][batch_idx, ...]

            trainer.logger.experiment.add_text(
                f"zpk/{batch_idx+1}",
                self.build_zpk_table(z, p, k),
                trainer.global_step,
            )

    def build_zpk_table(self, z, p, k):
        table = "| sos | zeros | poles | gains | \n"
        table += "|-----|-------|-------|-------| \n"

        for n in range(z.shape[0]):
            table += f"|{n+1} | {z[n]:3.3f} | {p[n]:3.3f} | {k[n]:0.5f} | \n"

        return table


class LogTransferFnPlots(Callback):
    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        if outputs is not None and batch_idx == 0:
            pred_sos = outputs["pred_sos"]
            sos = outputs["sos"]

            trainer.logger.experiment.add_image(
                f"mag-grid/{batch_idx+1}",
                plot_response_grid(pred_sos, target_coefs=sos),
                trainer.global_step,
            )
