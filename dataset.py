from torch.utils.data import Dataset, DataLoader
from transforms import get_transforms
from PIL import Image

class FaceCupDataset(Dataset):
    def __init__(self, df, get_x, trans):
        super().__init__()
        self.df = df
        self.x_col = get_x
        self.transform = trans

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        img = Image.open(self.df.loc[idx, self.x_col])
        img_tensor = self.transform(img)

        return img_tensor


def get_image_dataloader(df, get_x='file_path', batch_size=1, shuffle=False, num_workers=0):
    """
    Provides an iterable over the given dataset.

    :param df (pd.DataFrame): dataframe from which to load the data.
    :param get_x (str): column name in given dataframe which contains file path to images.
    :param batch_size (int): how many samples per batch to load (default: 1).
    :param shuffle (boolean): set to True to have the data reshuffled at every epoch (default: False).
    :param num_workers (int): how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.

    :return: torch.utils.data.DataLoader
    """
    dataset = FaceCupDataset(df=df, get_x=get_x, trans=get_transforms())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
