# Source: https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ
# References:
    # https://github.com/KimRass/DCGAN/blob/main/celeba.py

from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split


class CelebADataset(Dataset):
    def __init__(self, data_dir, img_size, hflip):
        super().__init__()

        self.img_paths = sorted(list(Path(data_dir).glob("**/*.jpg")))

        transforms = [
            A.SmallestMaxSize(max_size=img_size),
            A.CenterCrop(height=img_size, width=img_size),
            # "We assume that image data consists of integers in $\{0, 1, \ldots, 255\}$ scaled linearly
            # to $[-1, 1]$. This ensures that the neural network reverse process operates
            # on consistently scaled inputs starting from the standard normal prior $p(x_{T})$."
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
        if hflip:
            # "We used random horizontal flips during training. We found flips to improve sample quality slightly."
            transforms.append(A.HorizontalFlip(p=0.5))
        transforms.append(ToTensorV2())
        self.transforms = A.Compose(transforms)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.img_paths[idx]), flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transforms(image=img)["image"]


def dses_to_dls(train_ds, val_ds, test_ds, batch_size, n_cpus):
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        num_workers=n_cpus,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        num_workers=n_cpus,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        drop_last=True,
        persistent_workers=False,
        num_workers=n_cpus,
    )
    return train_dl, val_dl, test_dl


def get_dls(data_dir, img_size, batch_size, n_cpus, seed, val_ratio=0.2, test_ratio=0.1):
    train_val_test_ds = CelebADataset(data_dir=data_dir, img_size=img_size, hflip=True)
    train_val_idx, test_idx = train_test_split(
        np.arange(len(train_val_test_ds)),
        test_size=test_ratio,
        random_state=seed,
        shuffle=True,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
    )
    train_ds = Subset(train_val_test_ds, train_idx)
    # train_ds = Subset(train_val_test_ds, range(batch_size))
    val_ds = Subset(train_val_test_ds, val_idx)
    test_ds = Subset(train_val_test_ds, test_idx)
    return dses_to_dls(
        train_ds, val_ds, test_ds, batch_size=batch_size, n_cpus=n_cpus,
    )


class ImageGridDataset(Dataset):
    def __init__(self, data_dir, img_size, n_cells=100, padding=1):
        super().__init__()

        self.img_paths = sorted(list(Path(data_dir).glob("**/*.jpg")))
        self.img_size = img_size
        self.padding = padding
        self.n_cells = n_cells

        self.transformer = T.Compose(
            [T.ToTensor(), T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))],
        )

    def __len__(self):
        return len(self.img_paths) * self.n_cells

    def _idx_to_dimension(self, idx):
        return self.padding * (idx + 1) + self.img_size * idx

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx // self.n_cells]).convert("RGB")
        image = self.transformer(image)
        row_idx = (idx % self.n_cells) // int((self.n_cells ** 0.5))
        col_idx = (idx % self.n_cells) % int((self.n_cells ** 0.5))
        return image[
            :,
            self._idx_to_dimension(row_idx): self._idx_to_dimension(row_idx) + self.img_size,
            self._idx_to_dimension(col_idx): self._idx_to_dimension(col_idx) + self.img_size,
        ]
