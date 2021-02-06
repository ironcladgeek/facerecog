from utils import *
from add_overlay import add_face_mask
from transforms import augment_images
from embeddings import get_embeddings
import time

def get_embeddings_df(src_dir, dst_dir=None, add_mask=False, do_augmentation=False):
    # TODO: supply docstring
    s = time.time()
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir) if dst_dir is not None else src_dir
    # create destination directory if not exists
    dst_dir.mkdir(parents=True, exist_ok=True)

    if add_mask:
        add_face_mask(src_path=src_dir,
                      dst_path=dst_dir,
                      copy_originals_to_dst=True,
                      mask_colors=['white', 'black', 'blue', 'red'])
        src_dir = dst_dir

    crop_images(src_dir=src_dir, dst_dir=dst_dir)

    if do_augmentation:
        augment_images(src_dir=src_dir,
                       dst_dir=dst_dir,
                       h_flip=True,
                       clahe=True,
                       suffix='__augmented__')

    df = get_df_from_folder(src_dir=src_dir,
                            sort_by_name=True,
                            with_original_names=True)

    df = get_embeddings(df=df,
                        x_col='file_path',
                        pretrained_model='vggface2',
                        batch_size=16,
                        num_workers=0,
                        res_col_name='embedding')

    elapsed = format_time(time.time() - s)
    print(f'Total time: {elapsed}')
    return df
