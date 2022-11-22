import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from skimage import io

class ImageFolderVimeo(Dataset):
    def __init__(self, root, transform=None, split="train"):
        self.mode = split
        self.transform = transform
        self.samples = []
        split_dir = Path(root) / Path("vimeo_septuplet/sequences")
        for sub_f in split_dir.iterdir():
            if sub_f.is_dir():
                for sub_sub_f in Path(sub_f).iterdir():
                    self.samples += list(sub_sub_f.iterdir())

        if not split_dir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

    def __getitem__(self, index):
        img = io.imread(str(self.samples[index]))
        if self.transform:
            return self.transform(image=img)["image"]
        return img

    def __len__(self):
        return len(self.samples)


class Kodak24Dataset(Dataset):
    def __init__(self, root, transform=None, split="kodak24"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]
        self.transform = transform
        self.mode = split

    def __getitem__(self, index):
        img = io.imread(str(self.samples[index]))
        if self.transform:
            return self.transform(image=img)["image"]
        return img

    def __len__(self):
        return len(self.samples)


def build_dataset(args):
    train_transforms = A.Compose(
        [
            A.RandomCrop(args.block_size, args.block_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
            ToTensorV2(),
        ]
    )

    test_transforms = A.Compose(
        [
            A.RandomCrop(args.block_size, args.block_size),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
            ToTensorV2(),
        ]
    )

    train_dataset = ImageFolderVimeo(args.dataset, transform=train_transforms)
    test_dataset = Kodak24Dataset(args.dataset, transform=test_transforms)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=True,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader


def build_test_dataset(args):
    test_transforms = A.Compose(
        [
            A.RandomCrop(args.block_size, args.block_size),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
            ToTensorV2(),
        ]
    )

    test_dataset = Kodak24Dataset(args.dataset, transform=test_transforms)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
    )
    return test_dataloader
