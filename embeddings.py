import torch
from facenet_pytorch import InceptionResnetV1
from dataset import get_image_dataloader
import numpy as np
from tqdm import tqdm


def get_embeddings(df,
				   x_col='file_path',
				   pretrained_model='vggface2',
				   batch_size=16,
				   num_workers=2,
				   res_col_name='embedding',
				   use_saved_model=True):
    """
    Get face embeddings of dim=512 using pretrained model.

    :param df (pd.DataFrame): dataframe of images to be embedded.
    :param x_col (str): column name in given dataframe that contains images file paths.
    :param pretrained_model (str): name of the pre-trained model ('vggface2' or 'casia-webface').
    :param batch_size (int): how many samples per batch to load (default: 16).
    :param num_workers (int): how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
    :param res_col_name (str): face embeddings will be saved in this column.
    :param use_saved_model (boolean): If True, loads the saved model, otherwise downloads pretrained model from internet.

    :return (pd.DataFrame): original dataframe plus embeddings.
    """
    _df = df.copy()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Loading pre-trained model ...')
    if use_saved_model:
        if pretrained_model == 'vggface2':
            resnet = torch.load('models/resnet_vggface2.pth')
    else:
        resnet = InceptionResnetV1(pretrained=pretrained_model)
    resnet = resnet.eval().to(device) # move the model to appropriate device (CPU or GPU)
    dl = get_image_dataloader(df=_df, get_x=x_col, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print('Getting face embeddings ...')
    embedding_arr = []
    for images in tqdm(dl):
        images = images.to(device)      # move images to device
        embeddings = resnet(images)     # get embeddings using pretrained model
        embedding_arr.append(embeddings.detach().cpu().numpy())

    # convert embeddings to list and store in dataframe
    embedding_list = np.vstack(embedding_arr).tolist()
    _df[res_col_name] = embedding_list

    return _df
