from pytorch_lightning import Trainer
from torch.utils.data.dataloader import DataLoader
from dataset import NumberAdd
from model import Model1

# Data loaders
dl_train = DataLoader(NumberAdd(20000), batch_size=8, shuffle=True, num_workers=1)
dl_valid = DataLoader(NumberAdd(1000), batch_size=8, shuffle=True, num_workers=1)
dl_tests = DataLoader(NumberAdd(500), batch_size=8, shuffle=True, num_workers=1)

# Model
model = Model1()

# Trainer
trainer = Trainer(max_epochs=500, progress_bar_refresh_rate=20)
trainer.fit(model=model, train_dataloaders=dl_train, val_dataloaders=dl_valid)

