import subprocess
from tqdm import tqdm
from pathlib import Path
import glob
import shutil
from utils import get_image_files


def add_face_mask(src_path, mask_colors=['white', 'black', 'blue', 'red']):
    """Add face make to images if detect a face in the image. Nothing happens if no face detected in the image.
    Copy the original images plus mask-added ones into a new directory named <original-directory>_with_mask

    :param src_path (str, PosixPath): Path to directory of images
    :param mask_colors (list): List of mask colors to be added. Could be 'white', 'black', 'blue' and 'red'.
    """
    # check the color masks
    for color in mask_colors:
        if color not in ['white', 'black', 'blue', 'red']:
            raise NameError('Mask colors should be white, black, blue or red.')

    src_path = Path(src_path)
    path = src_path.absolute().parent / f'{src_path.name}_with_mask'
    shutil.copytree(src_path, path)         # copy original images into dest directory
    fp_images = get_image_files(path=path)  # get image files in the path

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

        print('Renaming images ...')
        # rename files
        for fp in tqdm(glob.iglob(str(path / '*-with-mask*'))):
            src = Path(fp)
            new_name = src.name.replace('-with-mask', f'__{color}-mask__')
            dst = path / f'{new_name}'
            shutil.move(src, dst)