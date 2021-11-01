import torch as T
from pytorch_lightning import LightningModule
from torch import nn
from torch import optim
from torch.nn.parameter import Parameter

from modules import Adder, Substracter


class Model1(LightningModule):
    def __init__(self, lr, optim_conf: dict={}):
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

        # Optimizer conf
        self.lr = lr
        self.optim_conf = optim_conf

    def forward(self, x):
        y0 = self.adder(x)
        y1 = self.substracter(x)
        y = (T.sigmoid(self.weights_adder) * y0 + 
             T.sigmoid(self.weights_substracter) * y1)
        return y

    def step(self, batch, batch_idx, *args, **kwargs) -> T.Tensor:
        # Unpacking
        samples, targets = batch

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

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=self.lr, **self.optim_conf)

    def on_epoch_end(self) -> None:
        self.log("weights/wa", self.weights_adder)
        self.log("weights/ws", self.weights_substracter)
        self.log("weights/sigma(wa)", T.sigmoid(self.weights_adder))
        self.log("weights/sigma(ws)", T.sigmoid(self.weights_substracter))