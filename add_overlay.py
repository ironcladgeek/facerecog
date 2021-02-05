import subprocess
from tqdm import tqdm
from pathlib import Path
import glob
import shutil
from utils import get_image_files


def add_face_mask(src_path, dst_path=None, copy_originals_to_dst=True, mask_colors=['white', 'black', 'blue', 'red']):
    """
    Add face make to images if detect a face in the image. Nothing happens if no face detected in the image.

    :param src_path (str, PosixPath): Path to directory of images.
    :param dst_path (str, PosixPath): If not None, the mask added images will be saved here, otherwise, in src_path.
    :param copy_originals_to_dst (boolean): If true, copy images found in src_path to dst_path.
    :param mask_colors (list[str]): List of mask colors to be added. Could be 'white', 'black', 'blue' and 'red'.
    """
    # check the color masks
    for color in mask_colors:
        if color not in ['white', 'black', 'blue', 'red']:
            raise NameError('Mask colors should be white, black, blue or red.')

    src_path = Path(src_path)
    dst_path = Path(dst_path) if dst_path is not None else src_path
    # create destination directory if not exists
    dst_path.mkdir(parents=True, exist_ok=True)
    # get image files in the src_path
    fp_images = get_image_files(path=src_path)


    for color in mask_colors:
        print(f'Adding {color} mask to images ...')
        # add mask to images
        for fp in tqdm(fp_images):
            if color == 'white':
                command = f"face-mask {fp}"
            else:
                command = f"face-mask {fp} --{color}"

            # execute the shell command in Python
            process = subprocess.Popen(command.split(),
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()

        # rename files
        for fp in glob.iglob(str(src_path / '*-with-mask*')):
            src = Path(fp)
            new_name = src.name.replace('-with-mask', f'__{color}-mask__')
            dst = dst_path / f'{new_name}'
            shutil.move(src, dst)

    # copy original images to dst_path
    if copy_originals_to_dst and src_path != dst_path:
        for fp in fp_images:
            dst = dst_path / fp.name
            shutil.copy(fp, dst)