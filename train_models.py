from dataset import ArtDataModule
from models import VitModule, RegNetModule
from models import TransferModule
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging, TQDMProgressBar
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning import Trainer
import torch
import warnings
warnings.filterwarnings('ignore')


EXP_NAME = "testing_224"
datamodule = ArtDataModule('train.csv', size=224, batch_size=8)

for name, model in zip(['VIT_32_layers', 'REGNET_y128gf'], [VitModule(), RegNetModule()]):
    print(f'[INFO] Training {name}')
    transfer_model =  TransferModule(model)

    trainer = Trainer(
        logger=MLFlowLogger(experiment_name=EXP_NAME, run_name=name, tracking_uri='file:./ml-runs', artifact_location='models'),
        callbacks=[TQDMProgressBar(10), ModelCheckpoint(dirpath='models', filename=f'{name}_model'), StochasticWeightAveraging(1e-3)],
        min_epochs=5, max_epochs=30, gpus=1, gradient_clip_val=0.1,
        auto_lr_find=True, auto_scale_batch_size=False,
        # fast_dev_run=True
    )
    trainer.tune(transfer_model, datamodule=datamodule)
    trainer.fit(transfer_model, datamodule=datamodule)
    torch.save(transfer_model.state_dict(), f'models/{name}/{name}_{EXP_NAME}.pth')