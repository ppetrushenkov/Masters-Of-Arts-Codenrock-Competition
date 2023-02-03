from torchvision.models import vit_l_32, ViT_L_32_Weights
from torchvision.models import regnet_y_128gf, RegNet_Y_128GF_Weights
# from torchvision.models import convnext_large, ConvNeXt_Large_Weights
from torchmetrics import Accuracy
from pytorch_lightning import LightningModule
import torch.nn as nn
import torch


class VitModule(nn.Module):
    """VIT Large model with 32 layers"""
    def __init__(self) -> None:
        super().__init__()
        self.model = vit_l_32(weights=ViT_L_32_Weights.IMAGENET1K_V1)
        for param in self.model.conv_proj.parameters():
            param.requires_grad = False
        for param in self.model.encoder.parameters():
            param.requires_grad = False

        self.model.heads = nn.Sequential(
            nn.Linear(1024, 1000), nn.LeakyReLU(),
            nn.Linear(1000, 512), nn.LeakyReLU(),
            nn.Linear(512, 40, bias=False)
        )

    def forward(self, x):
        return self.model(x)

class RegNetModule(nn.Module):
    """REGNET y128gf model"""
    def __init__(self) -> None:
        super().__init__()
        self.model = regnet_y_128gf(weights=RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1)
        for param in self.model.stem.parameters():
            param.requires_grad = False
        for param in self.model.trunk_output.parameters():
            param.requires_grad = False
        
        self.model.fc = nn.Sequential(
            nn.Linear(7392, 1000), nn.LeakyReLU(),
            nn.Linear(1000, 512), nn.LeakyReLU(),
            nn.Linear(512, 40)
        )

    def forward(self, x):
        return self.model(x)


class TransferModule(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.acc = Accuracy('multiclass', num_classes=40)
        self.lr = 3e-5

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        predictions = self.model(images)
        loss = nn.functional.cross_entropy(predictions, labels, label_smoothing=0.1)
        acc = self.acc(predictions, labels)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        predictions = self.model(images)
        loss = nn.functional.cross_entropy(predictions, labels)
        acc = self.acc(predictions, labels)
        self.log('valid_loss', loss, prog_bar=True)
        self.log('valid_acc', acc, prog_bar=True)
        return acc

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, amsgrad=True)
