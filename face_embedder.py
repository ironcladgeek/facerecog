import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.notebook import tqdm
import joblib
import re
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class FaceCupDataset(Dataset):
    def __init__(self, df, trans):
        super().__init__()
        self.df = df
        self.transform = trans

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        img = Image.open(self.df.loc[idx, 'file_path'])
        img_tensor = self.transform(img)

        return img_tensor


class Embedder():
    def __init__(self):

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.trans = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            fixed_image_standardization
        ])
        self.resnet_v = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def embedder(self, image_file_path):
        fp_images = sorted([fp for fp in image_file_path.iterdir()])
        fnames = list(map(lambda x: x.name, fp_images))
        df = pd.DataFrame(zip(fnames, fp_images), columns=['image_id', 'file_path'])

        trainset = FaceCupDataset(df, self.trans)
        trainloader = DataLoader(trainset, batch_size=64, shuffle=False, num_workers=4)

        embedding_arr_v = []

        for images in tqdm(trainloader):
            images = images.to(self.device)      
            embeddings_v = self.resnet_v(images)

            embedding_arr_v.append(embeddings_v.detach().cpu().numpy())


        embedding_arr_v = np.vstack(embedding_arr_v)
        embedding_lst_v = embedding_arr_v.tolist()

        df['embeddings_v'] = embedding_lst_v

        return df