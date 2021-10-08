import os
import glob
import wget
import torch
import zipfile
import scipy.signal
from iirnet.mlp import MLPModel


class Designer:
    def __init__(self):
        # assume checkpoints are stored in logs/
        if not os.path.isdir(os.path.join("logs", "filter_order")):
            print("No checkpoints found. Downloading...")
            if not os.path.isdir("logs"):
                os.makedirs("logs")
            zippath = wget.download(
                "https://zenodo.org/record/5550275/files/filter_order.zip",
                out="logs/",
            )
            print("Done.")
            with zipfile.ZipFile(zippath, "r") as zip_ref:
                zip_ref.extractall("logs")
            print("Extracted.")
        # get all the checkpoints for each order
        ckpt_dirs = glob.glob(os.path.join("logs", "filter_order", "*"))
        ckpt_dirs = [c for c in ckpt_dirs if os.path.isdir(c)]

        # load the models
        self.models = {}
        for ckpt_dir in ckpt_dirs:
            search_path = os.path.join(
                ckpt_dir,
                "lightning_logs",
                "version_0",
                "checkpoints",
                "*.ckpt",
            )
            ckpt_path = glob.glob(search_path)[0]
            order = int(os.path.basename(ckpt_dir).split("_")[2].split("=")[-1])
            model = MLPModel.load_from_checkpoint(ckpt_path)
            model.eval()
            self.models[order] = model

    def __call__(
        self,
        n: int,
        m: list,
        mode: str = "linear",
        output: str = "sos",
    ):

        # check the order is valid
        if n not in [4, 8, 16, 32, 64]:
            raise ValueError(f"Invalid order n = {n}. Must be in: [4, 8, 16, 32, 64]")

        # create target tensor and check shape
        m = torch.tensor(m).float()
        if m.ndim > 1:
            raise ValueError(f"m must have only one dimension. Found {m.ndim}.")

        m = m.view(1, 1, -1)

        # interpolate the magnitude specification to fit 512
        if m.shape[-1] != 512:
            m_int = torch.nn.functional.interpolate(m, 512, mode=mode)
        else:
            m_int = m

        # normalize the target response
        m_int = m_int.clamp(-128.0, 128.0)
        m_int /= 128

        # call the model to estimate filter parameters
        with torch.no_grad():
            sos, zpk = self.models[n](m_int)
            sos = sos.squeeze(0)

        # return in the desired format
        if output == "sos":
            return sos

        elif output == "ba":
            ba = scipy.signal.sos2tf(sos)
            return ba

        else:
            raise ValueError(f"Invalid output: {output}. Must be 'sos' or 'ba'. ")
