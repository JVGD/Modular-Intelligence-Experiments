from pytorch_lightning import Trainer
from torch.utils.data.dataloader import DataLoader
from dataset import NumberAdd
from model import Model1

# Data loaders
dl_train = DataLoader(NumberAdd(20000), batch_size=8, shuffle=True, num_workers=0)
dl_valid = DataLoader(NumberAdd(1000), batch_size=8, shuffle=True, num_workers=0)
dl_tests = DataLoader(NumberAdd(500), batch_size=8, shuffle=True, num_workers=0)

# Model
model = Model1(lr=1e-4, optim_conf={"momentum": 0.9})

# Trainer
trainer = Trainer(max_epochs=500)
trainer.fit(model=model, train_dataloaders=dl_train, val_dataloaders=dl_valid)

