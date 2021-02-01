from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN as mtcnn

def get_image_files(path):
    path = Path(path)
    image_extensions = ['.jpg', '.png', '.jpeg']
    return [x for x in path.iterdir() if x.suffix.lower() in image_extensions]


def crop(srcDir):

    print("start cropping faces ...")
    print("\n")
    src_path = Path(srcDir)
    save_path = src_path.parent / Path(f'{src_path.name}_cropped')
    save_path.mkdir(parents=True, exist_ok=True) ## create path if it doesn't exist
    fp_images = get_image_files(path=src_path)  # get image files in the path
    for fp in tqdm(fp_images):
        try:
            img = Image.open(str(fp)) # load the image
            # detect the face and save the cropped image
            img_crp = mtcnn(img, save_path=save_path / Path(fp.name.strip(".jpg") + "cropped__.jpg"))
        except TypeError:

            # resize the image
            img = img.resize((320, 320))

            width, height = img.size   # Get dimensions
            new_width, new_height = 160, 160
            left = (width - new_width)/2
            top = (height - new_height)/2
            right = (width + new_width)/2
            bottom = (height + new_height)/2

            # CenterCrop the image
            img = img.crop((left, top, right, bottom))

            # save the image
            img.save(fp=save_path / Path(fp.name.strip(".jpg") + "cropped__.jpg"))
