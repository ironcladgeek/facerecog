from utils import *
from add_overlay import add_face_mask
from transforms import augment_images
from embeddings import get_embeddings
from similarity import build_index, assessor
import time
import shutil
from tqdm import tqdm
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np


def producer(src_dir,
             dst_dir=None,
             do_masking=False,
             mask_colors=['white', 'blue', 'black'],
             do_augs=False,
             h_flip=True,
             clahe=True,
             batch_size=16,
             use_saved_model=True,
             copy_src_to_dst_dir=False,
             parallel=True):
    """
    Do image processing, augmentation and feature extraction for images in src_dir.

    :param src_dir (str, PosixPath): Source directory of images.
    :param dst_dir (str, PosixPath): If not None, the processed images will be saved here, otherwise, in src_dir.
    :param do_masking (boolean): If True, will add face masks to images. default False.
    :param mask_colors (list): List of face mask colors to be added to images.
    :param do_augs (boolean): If True, will do image augmentations. default False.
    :param h_flip (boolean): If True, apply horizontal flip augmentation on images.
    :param clahe (boolean): If True, apply CLAHE augmentation.
    :param batch_size (int): how many samples per batch to load (default: 16).
    :param use_saved_model (boolean): If True, loads the saved pre-trained model. Otherwise, downloads it from internet.
    :param copy_src_to_dst_dir (boolean): If True, copy images found in src_dir to dst_dir.
    :param parallel (boolean): If True, runs code using multi-processors.

    :return (pd.DataFrame): DataFrame contains of image names and embeddings.
    """
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

def aligner(gallery_df,
            probe_df,
            gallery_id_col='image_original_name',
            gallery_embedding_col='embedding',
            probe_id_col='image_original_name',
            probe_embedding_col='embedding'):
    """
    Get similarities of images in probe_df to all images in gallery_df.

    :param gallery_df (pd.DataFrame): DataFrame of gallery images embeddings.
    :param probe_df (pd.DataFrame): DataFrame of probe images embeddings
    :param gallery_id_col (str): Column name in gallery_df that contains image names.
    :param gallery_embedding_col (str): Column name in gallery_df that contains image embeddings.
    :param probe_id_col (str): Column name in probe_df that contains image names.
    :param probe_embedding_col (str): Column name in gallery_df that contains image embeddings.
    :return (dict): Dictionary of similarities. Keys are probe images.
        Values are sorted gallery images based on being most similar.
    """

    result = OrderedDict()
    annoy_index = build_index(gallery_df[gallery_embedding_col].values, metric='euclidean', n_trees=50)
    for id_probe, vec in tqdm(enumerate(probe_df[probe_embedding_col].values)):
        sim_idx, sim_distances = assessor(annoy_index, query_vec=vec, k=-1, include_distances=True)
        tmp_df = gallery_df[[gallery_id_col]].copy()
        tmp_df.loc[sim_idx, 'distance'] = sim_distances
        tmp_df.sort_values(by='distance', ascending=True, inplace=True)
        tmp_df.drop_duplicates(subset=gallery_id_col, keep='first', inplace=True)

        similarities = [{img_id: img_dist} for img_id, img_dist in zip(tmp_df[gallery_id_col].values,
                                                                       tmp_df['distance'].values)]
        result[probe_df.loc[id_probe, probe_id_col]] = similarities

    return result

def depictor(similarities, gallery_dir, probe_dir, fig_dir, add_suffix=True, suffix='.jpg'):
    """
    Generate pair plots for probe images and gallery images.

    :param similarities (dict): Dictionary of similarities.
    :param gallery_dir (str, PosixPath): Path to gallery images directory.
    :param probe_dir (str, PosixPath): Path to probe images directory.
    :param fig_dir (str, PosixPath): Path to output directory for saving plots. Will be created if not exists.
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    for fn_probe, sim_l in tqdm(similarities.items()):
        pfn = fn_probe + suffix if add_suffix else fn_probe
        probe_img = Image.open(os.path.join(probe_dir, pfn))
        fn_gallery = list(sim_l[0].keys())[0]
        gfn = fn_gallery + suffix if add_suffix else fn_gallery
        gallery_img = Image.open(os.path.join(gallery_dir, gfn))

        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.imshow(np.array(probe_img))
        ax1.title.set_text('Probe Image')
        ax2.imshow(np.array(gallery_img))
        ax2.title.set_text('Gallery Image')
        plt.savefig(str(fig_dir / f"{fn_probe.split('.')[0]}_{fn_gallery.split('.')[0]}.jpg"), dpi=100, bbox_inches='tight')
        plt.close()
