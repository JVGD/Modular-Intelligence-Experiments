import torch as T
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn.parameter import Parameter

from experiment1.modules import Adder, Substracter


class Model1(LightningModule):
    def __init__(self):
        """Modular AI approach 1"""
        super().__init__()

        # Usable modules
        self.adder = Adder()
        self.substracter = Substracter()

        # Model weights
        self.weights_adder = Parameter(data=T.rand(1), requires_grad=True)
        self.weights_substracter = Parameter(data=T.rand(1), requires_grad=True)

        # Loss
        self.criteria = nn.L1Loss()

    def forward(self, x):
        y0 = self.adder(x)
        y1 = self.substracter(x)
        y = self.weights_adder * y0 + self.weights_substracter * y1
        return y

    def step(self, batch, batch_idx, *args, **kwargs) -> T.Tensor:
        # Unpacking
        samples = batch["samples"]
        targets = batch["targets"]

        # Forward
        targets_pred = self(samples)

        # Loss
        loss = self.criteria(targets, targets_pred)
        return loss

    def training_step(self, batch, batch_idx, *args, **kwargs) -> T.Tensor:
        # Loss
        loss = self.step(batch, batch_idx, *args, *kwargs)
        self.log("loss/train", loss)
        return loss
        
    def validation_step(self, batch, batch_idx, *args, **kwargs) -> None:
        # Loss
        loss = self.step(batch, batch_idx, *args, *kwargs)
        self.log("loss/valid", loss)
