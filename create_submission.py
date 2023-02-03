from tqdm.auto import tqdm
from dataset import ArtDataset
from torch.utils.data import DataLoader
from torchvision import transforms as T
from models import TransferModule, VitModule
import torch
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def get_predictions(model, dataloader: DataLoader) -> list:
    """Return prediction list for specified dataloader"""
    print('[INFO] Start making predictions...')
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            img, _ = batch
            img = img.to(DEVICE)
            _, preds = torch.max(model(img), 1)
            predictions.extend(preds.cpu().numpy().tolist())
    print('[INFO] Done')
    return predictions

def create_csv_submit(predictions: list, img_names: list, name:str='submission'):
    print('[INFO] Create submission file...')
    submit = pd.DataFrame(
        {'image_name': img_names,
        'label_id': predictions}
    )
    submit.to_csv(name + '.csv', index=False)
    print(f'[INFO] Done. File saved as {name}.csv')
    return

SIZE = 224
BATCH_SIZE = 32
IMG_PATH = 'test'
PATH2MODEL = 'models/VIT_32_layers/VIT_32_layers_testing_224_img_size_sgd.pth'
DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'

transform = T.Compose([
    T.Resize((SIZE, SIZE)),
    T.ToTensor(),
    T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

print('[INFO] Load the model...')
weights = torch.load(PATH2MODEL, map_location=DEVICE)
model = TransferModule(VitModule()).to(DEVICE)
model.load_state_dict(weights)
model.eval()
print('[INFO] Done')

dataset = ArtDataset(IMG_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

img_names = [item.split("/")[-1] for item in dataset.files]
predictions = get_predictions(model, dataloader)
create_csv_submit(predictions, img_names, name='vit_submission')