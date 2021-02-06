from pathlib import Path
from tqdm import tqdm
from PIL import Image
from facenet_pytorch import MTCNN


mtcnn = MTCNN(image_size=160)


def get_image_files(path):
    path = Path(path)
    image_extensions = ['.jpg', '.png', '.jpeg']
    return [x for x in path.iterdir() if x.suffix.lower() in image_extensions]


def face_crop(img, save_path=None):
    """
    Detect the face in the image and crop. If no face is detected, will do a CenterCrop.
    :param img (PILImage): Image to be cropped.
    :param save_path (str, PosixPath): Save path of the cropped image. default=None.
    :return: The cropped image.
    """
    try:
        return mtcnn(img, save_path=str(save_path))
    except TypeError:
        img_crp = center_crop(img)
        if save_path is not None:
            img_crp.save(fp=str(save_path))
        return img_crp


def center_crop(img, image_size=160, enlarge=True):
    """
    Do a center crop on received image.
    :param img (PILImage): Image to be cropped.
    :param image_size (int): Size of the cropped image. (default=160).
    :param enlarge (boolean): If true, double the size of the image.
    :return: The cropped image.
    """
    image_size = int(image_size)
    if enlarge:
        # double the image size
        img = img.resize((image_size*2, image_size*2))

    width, height = img.size  # Get dimensions
    new_width, new_height = image_size, image_size
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    # CenterCrop the image
    return img.crop((left, top, right, bottom))


def crop_images(src_dir, dst_dir=None):
    """
    Face crop the images found in the src_dir.
    :param src_dir (str, PosixPath): Source directory of images to be cropped.
    :param dst_dir (str, PosixPath): If not None, the cropped images will be saved here, otherwise, in src_dir.
    :return: None
    """
    print("Cropping images ...")

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir) if dst_dir is not None else src_dir
    # create destination directory if not exists
    dst_dir.mkdir(parents=True, exist_ok=True)

    fp_images = get_image_files(src_dir)
    for fp in tqdm(fp_images):
        img = Image.open(fp)
        save_path = str(dst_dir / fp.name)
        face_crop(img=img, save_path=save_path)
