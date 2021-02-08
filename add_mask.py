import time
import glob
from tqdm import tqdm
from facemasker import FaceMasker
def add_mask(src_path,dst_path, mask_colors):
    start = time.time()
    print("adding masks ...")
    for dir_ in tqdm(glob.iglob(src_path+"/*.jpg")):
        f = FaceMasker(src_path=dir_, mask_colors=mask_colors,dst_path="out")
        f.mask()
    end = time.time()
    print("time elapsed for putting mask = {}".format(end-start))



########## TEST ##############

add_mask(src_path="gallery", dst_path="out", mask_colors=["red", "blue", "white", "black"])
