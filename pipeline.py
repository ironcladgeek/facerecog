from utils import *
from add_overlay import add_face_mask
from transforms import augment_images
from embeddings import get_embeddings
import time
import shutil


def producer(src_dir,
             dst_dir=None,
             do_masking=False,
             mask_colors=['blue', 'black'],
             do_augs=False,
             h_flip=True,
             clahe=True,
             batch_size=16,
             use_saved_model=True,
             copy_src_to_dst_dir=False,
             parallel=True):
    # TODO: supply docstring
    s = time.time()

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir) if dst_dir is not None else src_dir
    if copy_src_to_dst_dir and dst_dir is not None:
        shutil.copytree(src_dir, dst_dir)
        src_dir = dst_dir
    else:
        # create destination directory if not exists
        dst_dir.mkdir(parents=True, exist_ok=True)

    if do_masking:
        add_face_mask(src_path=src_dir,
                      dst_path=dst_dir,
                      copy_originals_to_dst=True,
                      mask_colors=mask_colors,
                      parallel=parallel)
        src_dir = dst_dir

    crop_images(src_dir=src_dir, dst_dir=dst_dir)
    src_dir = dst_dir

    if do_augs:
        augment_images(src_dir=src_dir,
                       dst_dir=dst_dir,
                       h_flip=h_flip,
                       clahe=clahe,
                       suffix='__augmented__')

    df = df_from_image_folder(src_dir=src_dir,
                              sort_by_name=True,
                              with_original_names=True)

    df = get_embeddings(df=df,
                        x_col='file_path',
                        pretrained_model='vggface2',
                        batch_size=batch_size,
                        num_workers=0,
                        res_col_name='embedding', 
                        use_saved_model=use_saved_model)

    elapsed = format_time(time.time() - s)
    print(f'Total time: {elapsed}')
    return df
