import os
import sys

# Add paths
current_dir = os.path.abspath(os.path.dirname(__file__))
rsbench_rss_path = os.path.join(current_dir, "rsbench-code", "rsseval", "rss")
sys.path.append(rsbench_rss_path)

from datasets.shortcutmnist import SHORTMNIST
import torch

class MockArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

ds_args = MockArgs(
    model="mnistnn",
    task="addition",
    c_sup=0,
    which_c=-1,
    batch_size=4,
    dataset="shortmnist",
    joint=False,
    backbone="none",
    n_epochs=1,
    lr=1e-3,
    exp_decay=0.99,
    warmup_steps=0,
    finetuning=False,
    validate=False,
)

dataset = SHORTMNIST(ds_args)
train, _, _ = dataset.get_data_loaders()

for x, y, c in train:
    print("Targets (sums):", y)
    print("Parity modulo 2:", y % 2)
    print("Concepts:", c)
    break
