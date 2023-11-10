# Source: https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ
# Reference: https://github.com/KimRass/DCGAN/blob/main/celeba.py

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from pathlib import Path


def get_transformer(img_size, hflip=False):
    transforms = [
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    if hflip:
        # "We used random horizontal flips during training. We found flips to improve sample quality slightly."
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class CelebADataset(Dataset):
    def __init__(self, data_dir, img_size):
        super().__init__()

        self.img_paths = list(Path(data_dir).glob("**/*.jpg"))
        self.img_size = img_size

        self.transformer = get_transformer(img_size=img_size, hflip=True)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        image = self.transformer(image)
        return image


def get_celeba_dataloader(data_dir, img_size, batch_size, n_workers):
    ds = CelebADataset(data_dir=data_dir, img_size=img_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=n_workers, drop_last=True)
    return dl
