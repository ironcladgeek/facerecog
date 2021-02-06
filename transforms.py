import numpy as np
import albumentations as A
from utils import get_image_files
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from torchvision import transforms
from facenet_pytorch import fixed_image_standardization


def get_augs(h_flip=True, clahe=False, clahe_clip_limit=(1, 4)):
    augs = []
    if h_flip:
        augs.append(A.HorizontalFlip(p=1.0))
    if clahe:
        augs.append(A.CLAHE(clip_limit=clahe_clip_limit, tile_grid_size=(5, 5), p=1.0))

    return A.Compose(augs)


def image_transform(img, augs):
    img = np.array(img)
    transformed = augs(image=np.array(img))
    return Image.fromarray(transformed['image'])


def augment_images(src_dir, dst_dir=None, h_flip=True, clahe=False, suffix='__augmented__'):
    """
    Apply image augmentations on images found in src_dir.

    :param src_dir (str, PosixPath): Source directory of images to be augmented.
    :param dst_dir (str, PosixPath): If not None, the augmented images will be saved here, otherwise, in src_dir.
    :param h_flip (boolean): If True, apply horizontal flip on images.
    :param clahe (boolean): If True, apply CLAHE (contrast limited adaptive histogram equalization) on images.
    :param suffix (str): suffix to be added to images filenames.
    """

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir) if dst_dir is not None else src_dir
    # create destination directory if not exists
    dst_dir.mkdir(parents=True, exist_ok=True)
    fp_images = get_image_files(path=src_dir)
    augs = get_augs(h_flip=h_flip, clahe=clahe, clahe_clip_limit=(1, 1))

    print('Applying augmentations ...')
    for fp in tqdm(fp_images):
        img = Image.open(fp)
        transformed_img = image_transform(img, augs)
        save_path = dst_dir / f"{fp.name.strip(fp.suffix)}{suffix}{fp.suffix}"
        transformed_img.save(str(save_path))


def get_transforms():
    """
    PyTorch transforms that apply standard normalizations on image and turns it to tensor.
    :return: torchvision.transforms
    """
    return transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])
