import numpy as np
from tqdm.notebook import tqdm
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN


class FaceDetector():
    def __init__(self):
        self.model = MTCNN()
        self.extensions = ['.jpg', '.png', '.jpeg']


    def crop_single_image(self, image_dir, cropped_image_dir):
        fp_images = [x for x in image_dir.iterdir() if x.suffix.lower() in self.extensions]
        try:
            fp = fp_images[0]
            img = Image.open(fp)
            save_path = str(cropped_image_dir / fp.name)
            img_crp = self.model(img, save_path=save_path)
        except TypeError:
            img = img.resize((320, 320))
            width, height = img.size

            new_width, new_height = 160, 160
            left = (width - new_width)/2
            top = (height - new_height)/2
            right = (width + new_width)/2
            bottom = (height + new_height)/2

            img = img.crop((left, top, right, bottom))

            img.save(fp=save_path)

    def crop_multi_images(self, image_dir, cropped_image_dir):
        not_detected_images = []
        fp_images = [x for x in image_dir.iterdir() if x.suffix.lower() in self.extensions]
        for fp in tqdm(fp_images):
            try:
                img = Image.open(fp)
                save_path = str(cropped_image_dir / fp.name)
                img_crp = self.model(img, save_path=save_path)
            except TypeError:
                not_detected_images.append(fp.name)

                img = img.resize((320, 320))
                
                width, height = img.size  
                new_width, new_height = 160, 160
                left = (width - new_width)/2
                top = (height - new_height)/2
                right = (width + new_width)/2
                bottom = (height + new_height)/2

                img = img.crop((left, top, right, bottom))

                img.save(fp=save_path)
