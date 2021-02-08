from facemasker import FaceMasker
from tqdm import tqdm
from pathlib import Path
import shutil
from utils import get_image_files, format_time
from multiprocessing import Pool, cpu_count
import time


def put_mask(fp, dst_path, mask_colors):
    f = FaceMasker(src_path=str(fp), dst_path=str(dst_path), mask_colors=mask_colors)
    f.mask()


def add_face_mask(src_path,
                  dst_path=None,
                  copy_originals_to_dst=True,
                  mask_colors=['white', 'black', 'blue', 'red'],
                  parallel=True):
    """
    Add face make to images if detect a face in the image. Nothing happens if no face detected in the image.

    :param src_path (str, PosixPath): Path to directory of images.
    :param dst_path (str, PosixPath): If not None, the mask added images will be saved here, otherwise, in src_path.
    :param copy_originals_to_dst (boolean): If true, copy images found in src_path to dst_path.
    :param mask_colors (list[str]): List of mask colors to be added. Could be 'white', 'black', 'blue' and 'red'.
    :param parallel (boolean): If True, add masks to images in parallel using multiprocessing.
    """
    # check the color masks
    s = time.time()
    for color in mask_colors:
        if color not in ['white', 'black', 'blue', 'red']:
            raise NameError('Mask colors should be white, black, blue or red.')

    src_path = Path(src_path)
    dst_path = Path(dst_path) if dst_path is not None else src_path
    # create destination directory if not exists
    dst_path.mkdir(parents=True, exist_ok=True)
    # get image files in the src_path
    fp_images = get_image_files(path=src_path)

    print('Adding face masks to images ...')
    if parallel:  # multiprocessing
        with Pool(processes=cpu_count()) as p:
            p.starmap(put_mask, [(fp, dst_path, mask_colors) for fp in fp_images])
    else:  # single process
        for fp in tqdm(fp_images):
            put_mask(fp, dst_path, mask_colors)

    # copy original images to dst_path
    if copy_originals_to_dst and src_path != dst_path:
        for fp in fp_images:
            dst = dst_path / fp.name
            shutil.copy(fp, dst)

    elapsed = format_time(time.time() - s)
    print(f'Adding masks took: {elapsed}')
