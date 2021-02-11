from pathlib import Path
from tqdm import tqdm
from PIL import Image
from facenet_pytorch import MTCNN
import pandas as pd
import re
import datetime

size = 160
mtcnn = MTCNN(image_size=size)


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
        img_crp = center_crop(img, image_size=size)
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
    print("Detecting faces and cropping images ...")

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir) if dst_dir is not None else src_dir
    # create destination directory if not exists
    dst_dir.mkdir(parents=True, exist_ok=True)

    fp_images = get_image_files(src_dir)
    for fp in tqdm(fp_images):
        img = Image.open(fp)
        save_path = str(dst_dir / fp.name)
        face_crop(img=img, save_path=save_path)


def df_from_image_folder(src_dir, sort_by_name=True, with_original_names=True):
    """
    Create a pd.DataFrame from image files found in the src_dir.

    :param src_dir (str, PosixPath): Source directory of images.
    :param sort_by_name (boolean): If True, the resulting dataframe will be sorted by image file names.
    :param with_original_names (boolean): If True, the resulting dataframe will have a column
        that contains original image names without '__.*__' pattern.

    :return (pd.DataFrame): df[['image_id', 'file_path', 'image_original_name']]
    """
    fp_images = get_image_files(path=src_dir)
    if sort_by_name:
        fp_images = sorted(fp_images)

    f_names = list(map(lambda x: x.name, fp_images))
    df =  pd.DataFrame(zip(f_names, fp_images), columns=['image_id', 'file_path'])

    if with_original_names:
        pat = r'.*?(__.*__)\..*'
        def remove_extra(image_name):
            image_name = str(image_name)
            res = re.findall(pat, image_name)
            if len(res) > 0:
                image_name = image_name.replace(res[0], '')
            return image_name.split('.')[0]     # remove file extension

        df['image_original_name'] = df['image_id'].apply(lambda x: remove_extra(x))

    return df


def format_time(elp):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round(elp))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

import pdb
def accuracy_score(similarities_dict, y_true_df):
    # TODO: supply docstring
    probe_images = [k for k, v in similarities_dict.items()]
    g_images = [v[:5] for k, v in similarities_dict.items()]
    gallery_images = []
    for o in g_images:
        gallery_images.append([k for d in o for k, v in d.items()])

    if isinstance(y_true_df, str):
        y_true_df = pd.read_csv(y_true_df)

    preds_df = pd.DataFrame(zip(probe_images, gallery_images), columns=['probe', 'gallery_top_5'])
    n = preds_df.shape[0]
    preds_df.index = preds_df['probe']
    preds_df = preds_df.reindex(y_true_df['probe'])
    preds_df.reset_index(drop=True, inplace=True)
    preds_df = preds_df.iloc[:n].copy()
    y_true_df = y_true_df.iloc[:n].copy()

    preds_df['y_true'] = y_true_df['gallery']

    preds_df['gallery_top_1'] = preds_df['gallery_top_5'].apply(lambda x: x[0])

    preds_df['is_top_1'] = (preds_df['gallery_top_1'] == preds_df['y_true'])

    preds_df['is_in_top_5'] = preds_df.apply(lambda x: x['y_true'] in x['gallery_top_5'], axis=1)

    acc_top_1 = preds_df['is_top_1'].sum() / preds_df.shape[0]
    acc_top_5 = preds_df['is_in_top_5'].sum() / preds_df.shape[0]

    return acc_top_1, acc_top_5, preds_df

