import torch

from iirnet.system import System
from iirnet.mlp import MLPModel

if __name__ == "__main__":

    model_ckpt = ""
    system = System.load_from_checkpoint(
        "logs/hidden_dim/epochs=500_filter-method=all_filter-order=32_hidden-dim=1024/lightning_logs/version_15/checkpoints/all-epoch=13-step=5474.ckpt",
        map_location="cpu",
    )
    model = system.model
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, f"traced_iirnet.pt")
