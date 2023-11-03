# Source: https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ
# Reference: https://github.com/KimRass/DCGAN/blob/main/celeba.py

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from pathlib import Path


class CelebADataset(Dataset):
    def __init__(self, data_dir, img_size, mean, std):
        super().__init__()

        self.img_paths = list(Path(data_dir).glob("**/*.jpg"))
        self.img_size = img_size

        self.transformer = T.Compose([
            T.Resize(img_size),
            T.CenterCrop(img_size),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            # "No pre-processing was applied to training images besides scaling to the range of the tanh
            # activation function $[-1, 1]$."
            T.Normalize(mean, std),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        image = self.transformer(image)
        return image


def get_celeba_dataloader(data_dir, img_size, mean, std, batch_size, n_workers):
    ds = CelebADataset(data_dir=data_dir, img_size=img_size, mean=mean, std=std)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=n_workers, drop_last=True)
    return dl
