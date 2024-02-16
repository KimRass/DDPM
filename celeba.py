# Source: https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ
# References:
    # https://github.com/KimRass/DCGAN/blob/main/celeba.py

from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
from pathlib import Path


def get_transformer(img_size):
    transforms = [
        T.Resize(img_size),
        T.CenterCrop(img_size),
        # "We used random horizontal flips during training. We found flips to improve sample quality slightly."
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        # "We assume that image data consists of integers in $\{0, 1, \ldots, 255\}$ scaled linearly
        # to $[-1, 1]$. This ensures that the neural network reverse process operates
        # on consistently scaled inputs starting from the standard normal prior $p(x_{T})$."
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    return T.Compose(transforms)


class CelebADataset(Dataset):
    def __init__(self, data_dir, img_size):
        super().__init__()

        self.img_paths = sorted(list(Path(data_dir).glob("**/*.jpg")))
        self.img_size = img_size

        self.transformer = get_transformer(img_size=img_size, hflip=True)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        image = self.transformer(image)
        return image


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
