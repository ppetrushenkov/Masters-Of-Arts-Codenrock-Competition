from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pytorch_lightning import LightningDataModule
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms as T
import os
import pandas as pd


class ArtDataset(Dataset):
    def __init__(self, root_dir, data: pd.DataFrame=None, transform=None):

        self.transform = transform
        self.files = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)]
        self.targets = None
        if data is not None:
            self.targets = data["label_id"].tolist()
            self.files = [os.path.join(root_dir, fname) for fname in data["image_name"].tolist()]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(self.files[idx]).convert('RGB')
        target = self.targets[idx] if self.targets else -1
        if self.transform:
            image = self.transform(image)
        return image, target


class ArtDataModule(LightningDataModule):
    def __init__(self, csv_path=None, batch_size:int=32, size:int=384) -> None:
        super().__init__()
        self.size = size
        self.csv_path = csv_path
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        df = pd.read_csv(self.csv_path, sep="\t")
        self.train, self.valid = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label_id'])
        self.sampler = self.random_sampler_weights(self.train)
        self.train_transform = T.Compose([
            T.RandomResizedCrop((self.size, self.size)),
            T.RandomChoice([T.TrivialAugmentWide(), T.AugMix()]),
            T.ToTensor(),
            T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
            T.RandomErasing(p=.1)
        ])
        self.valid_transform = T.Compose([
            T.Resize((self.size, self.size)),
            T.ToTensor(),
            T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])
        
    def setup(self, stage: str=None) -> None:
        self.train_dataset = ArtDataset('train', self.train, transform=self.train_transform)
        self.valid_dataset = ArtDataset('train', self.valid, transform=self.valid_transform)

    def random_sampler_weights(self, data: pd.DataFrame):
        count_per_class = data.label_id.value_counts(normalize=True)
        weights = 1 / count_per_class[data.label_id]
        return WeightedRandomSampler(weights=weights.sort_index().to_numpy(), num_samples=len(weights))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, sampler=self.sampler, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, self.batch_size, num_workers=4, pin_memory=True)