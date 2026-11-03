"""Datasets and dataloaders for LLFDisc.

Two dataset flavours are provided:

* :class:`PairedDataset` -- paired low-light / normal-light images, for training
  and for reference-based evaluation (PSNR, SSIM, LPIPS).
* :class:`DatasetFromFolderEval` -- unpaired low-light images, for no-reference
  evaluation (NIQE) on benchmarks such as DICM, LIME, MEF, NPE and VV.
"""

import os

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def _list_images(directory):
    """Returns the sorted image file names directly inside ``directory``.

    Sorting matters for :class:`PairedDataset`: low-light and normal-light
    images are matched by position, so both listings must use the same order.
    """
    return sorted(
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(IMAGE_EXTENSIONS)
    )


class PairedDataset(Dataset):
    """Paired low-light / normal-light image dataset.

    The two directories are listed and sorted independently, then paired by
    index, so corresponding images must share a naming scheme that sorts
    identically.

    Args:
        low_dir: Directory holding the low-light inputs.
        high_dir: Directory holding the normal-light targets.
        transform: Transform applied to both images (typically ``ToTensor``).
        crop_size: Side length of the random square crop. Only used when
            ``training`` is true.
        training: If true, apply the random crop; otherwise return full images.
    """

    def __init__(self, low_dir, high_dir, transform=None, crop_size=None, training=True):
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.transform = transform
        self.crop_size = crop_size
        self.training = training

        self.low_images = _list_images(low_dir)
        self.high_images = _list_images(high_dir)

        assert len(self.low_images) == len(self.high_images), (
            f"Mismatch in number of images: {len(self.low_images)} in {low_dir} "
            f"vs {len(self.high_images)} in {high_dir}"
        )

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        low_image_path = os.path.join(self.low_dir, self.low_images[idx])
        high_image_path = os.path.join(self.high_dir, self.high_images[idx])

        low_image = Image.open(low_image_path).convert('RGB')
        high_image = Image.open(high_image_path).convert('RGB')

        if self.transform:
            low_image = self.transform(low_image)
            high_image = self.transform(high_image)

        if self.training and self.crop_size:
            # Draw the crop box once and apply it to both images so the pair
            # stays spatially aligned.
            i, j, h, w = transforms.RandomCrop.get_params(
                low_image, output_size=(self.crop_size, self.crop_size))
            low_image = transforms.functional.crop(low_image, i, j, h, w)
            high_image = transforms.functional.crop(high_image, i, j, h, w)

        return low_image, high_image


class DatasetFromFolderEval(Dataset):
    """Unpaired low-light dataset for no-reference evaluation.

    Args:
        low_dir: Directory holding the low-light images.
        transform: Transform applied to each image (typically ``ToTensor``).

    Returns:
        Each item is a ``(image_tensor, file_name)`` tuple; the file name lets
        the caller save results under the original name.
    """

    def __init__(self, low_dir, transform=None):
        self.low_dir = low_dir
        self.transform = transform
        self.image_names = _list_images(low_dir)

        if not self.image_names:
            raise FileNotFoundError(f"No images found in {low_dir}")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        file_name = self.image_names[idx]
        image = Image.open(os.path.join(self.low_dir, file_name)).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, file_name


def create_dataloaders(train_low, train_high, test_low, test_high,
                       crop_size=256, batch_size=1, num_workers=4):
    """Builds the training and test dataloaders.

    Either pair of directories may be ``None``, in which case the corresponding
    loader is returned as ``None`` -- useful for evaluation-only scripts.

    Args:
        train_low: Low-light training directory, or ``None``.
        train_high: Normal-light training directory, or ``None``.
        test_low: Low-light test directory, or ``None``.
        test_high: Normal-light test directory, or ``None``.
        crop_size: Random crop size used for training.
        batch_size: Training batch size. The test loader always uses 1, since
            test images are not cropped and may differ in resolution.
        num_workers: Dataloader worker processes.

    Returns:
        Tuple ``(train_loader, test_loader)``.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader = None
    test_loader = None

    if train_low and train_high:
        train_dataset = PairedDataset(train_low, train_high, transform=transform,
                                      crop_size=crop_size, training=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers)

    if test_low and test_high:
        test_dataset = PairedDataset(test_low, test_high, transform=transform, training=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                 num_workers=num_workers)

    return train_loader, test_loader


def create_eval_dataloader(low_dir, num_workers=4):
    """Builds a batch-size-1 dataloader over an unpaired low-light directory.

    Args:
        low_dir: Directory holding the low-light images.
        num_workers: Dataloader worker processes.

    Returns:
        A ``DataLoader`` yielding ``(image_tensor, file_name)`` pairs.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = DatasetFromFolderEval(low_dir, transform=transform)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
