import torch
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from iirnet.data import IIRFilterDataset

num_points = 512
max_train_order = 16
num_train_examples = 1000
precompute = False


def wif(id):  # worker init function
    np.random.seed((id + torch.initial_seed()) % np.iinfo(np.int32).max)


# setup the dataloaders
train_datasetA = IIRFilterDataset(
    method="normal_poly",
    num_points=num_points,
    max_order=max_train_order,
    num_examples=num_train_examples,
    precompute=precompute,
)

train_datasetB = IIRFilterDataset(
    method="normal_biquad",
    num_points=num_points,
    max_order=max_train_order,
    num_examples=num_train_examples,
    precompute=precompute,
)

train_datasetC = IIRFilterDataset(
    method="uniform_disk",
    num_points=num_points,
    max_order=max_train_order,
    num_examples=num_train_examples,
    precompute=precompute,
)

train_datasetD = IIRFilterDataset(
    method="uniform_mag_disk",
    num_points=num_points,
    max_order=max_train_order,
    num_examples=num_train_examples,
    precompute=precompute,
)

train_datasetE = IIRFilterDataset(
    method="char_poly",
    num_points=num_points,
    max_order=max_train_order,
    num_examples=num_train_examples,
    precompute=precompute,
)

train_datasetF = IIRFilterDataset(
    method="uniform_parametric",
    num_points=num_points,
    max_order=max_train_order,
    num_examples=num_train_examples,
    precompute=precompute,
)

filter_datasets = {
    "normal_poly": train_datasetA,
    "normal_biquad": train_datasetB,
    "uniform_disk": train_datasetC,
    "uniform_mag_disk": train_datasetD,
    "char_poly": train_datasetE,
    "uniform_parametric": train_datasetF,
    "all": torch.utils.data.ConcatDataset(
        [
            train_datasetA,
            train_datasetB,
            train_datasetC,
            train_datasetD,
            train_datasetE,
            train_datasetF,
        ]
    ),
}

for filter_method, filter_dataset in filter_datasets.items():

    ranges = []
    means = []
    train_dataloader = torch.utils.data.DataLoader(
        filter_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=4,
        worker_init_fn=wif,
        pin_memory=True,
    )

    for bidx, batch in enumerate(train_dataloader, 0):
        mag, phs, real, imag, sos = batch
        # print(f"{filter_method}: {mag.min()} {mag.max()}")
        ranges.append(mag.min())
        ranges.append(mag.max())
        means.append(mag.mean())

        if mag.isnan().any():
            print(f"{filter_method}: NaN")

    print(
        f"{filter_method} |  min: {np.min(ranges):0.3f}  max: {np.max(ranges):0.3f}  mean: {np.mean(means)}"
    )
