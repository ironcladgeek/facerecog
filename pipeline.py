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
             mask_colors=['white', 'blue', 'black', 'red'],
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

def aligner(gallery_df,
            probe_df,
            gallery_id_col='image_original_name',
            gallery_embedding_col='embedding',
            probe_id_col='image_original_name',
            probe_embedding_col='embedding'):
    #TODO: supply docstring

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

def depictor(similarities, gallery_dir, probe_dir, fig_dir):
    # TODO: supply docstring
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    for fn_probe, sim_l in tqdm(similarities.items()):
        probe_img = Image.open(os.path.join(probe_dir, fn_probe))
        fn_gallery = list(sim_l[0].keys())[0]
        gallery_img = Image.open(os.path.join(gallery_dir, fn_gallery))

        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.imshow(np.array(probe_img))
        ax1.title.set_text('Probe Image')
        ax2.imshow(np.array(gallery_img))
        ax2.title.set_text('Gallery Image')
        plt.savefig(str(fig_dir / f"{fn_probe.split('.')[0]}_{fn_gallery.split('.')[0]}.jpg"), dpi=100, bbox_inches='tight')
        plt.close()
