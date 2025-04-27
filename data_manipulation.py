import h5py
import tensorflow as tf 
import torch
from torch.utils.data import Dataset

def make_h5_generator(x_path, y_path):
    def _gen():
        with h5py.File(x_path, 'r') as fx, h5py.File(y_path, 'r') as fy:
            x_ds = fx['x']
            y_ds = fy['y']
            for i in range(x_ds.shape[0]):
                img = x_ds[i]            
                lbl = y_ds[i].squeeze()  
                yield img, lbl
    return _gen

def create_data_pipeline(file_x, file_y, batch_size):
    AUTOTUNE = tf.data.AUTOTUNE
    ds_data = tf.data.Dataset.from_generator(
        make_h5_generator(file_x, file_y),
        output_types=(tf.uint8, tf.uint8),
        output_shapes=((96, 96, 3), ())
    ).map(lambda img, lbl: (tf.cast(img, tf.float32)/255.0, lbl),
        num_parallel_calls=AUTOTUNE
    ).batch(batch_size).prefetch(AUTOTUNE)
    return ds_data

class PatchCamelyonH5Dataset(Dataset):
    def __init__(self, x_path, y_path, transform=None):
        self.x_path = x_path
        self.y_path = y_path
        self.transform = transform

        with h5py.File(self.x_path, "r") as fx:
            self.len = fx["x"].shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        with h5py.File(self.x_path, "r") as fx, h5py.File(self.y_path, "r") as fy:
            img = fx["x"][idx] # uint8 H×W×3
            lbl = fy["y"][idx].squeeze()

        img = torch.from_numpy(img).float() / 255.0 # 0-1
        img = img.permute(2, 0, 1) # CHW

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(lbl, dtype=torch.long)