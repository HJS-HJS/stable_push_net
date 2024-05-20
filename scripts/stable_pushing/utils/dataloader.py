import os
import re
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from utils.dataloader_parallel import DataLoaderParallel

class PushNetDataset(Dataset):
    def __init__(self, dataset_dir: str, type: str='train', image_type: str='masked_image', num_debug_samples: int=0, zero_padding: int=7, pin_memory: bool=True):
        """Dataset class for DexNet.

        Args:
            dataset_dir (str): Path to the dataset directory.
            type (str, optional): Type of the dataset. Defaults to 'train'.
            image_type (str, optional): type of the network input image. Defaults to 'masked'.
            num_debug_samples (int, optional): _description_. Defaults to 0.
            zero_padding (int, optional): Data name padding number. Defaults to 7.
        """
        # dataset option
        self.image_type = image_type
        self.FILE_ZERO_PADDING_NUM = zero_padding
        self.pin_memory = pin_memory
        dataset_dir = os.path.expanduser('~') + '/' + dataset_dir

        # data directory
        self.tensor_dir = os.path.join(dataset_dir, 'tensors')
        split_dir  = os.path.join(dataset_dir, 'split')
        stats_dir  = os.path.join(dataset_dir, 'data_stats')
    
        # load indicies
        indices_file = os.path.join(split_dir, type + '_indices.npy')
        self.indices = np.load(indices_file)
        
        file_list = os.listdir(self.tensor_dir)
        file_list = [file_name for file_name in file_list if file_name.startswith('image')]
        indices = [int(re.findall(r'\d+', file_name)[0]) for file_name in file_list]
        indices = np.sort(indices)
        max_index = indices[-1]
        
        # Data normalization
        self.velocity_mean = np.load(os.path.join(stats_dir, 'velocity_mean.npy'))
        self.velocity_std  = np.load(os.path.join(stats_dir, 'velocity_std.npy'))
        data_loader_parallel = DataLoaderParallel(max_index, self.tensor_dir, self.FILE_ZERO_PADDING_NUM)
        
        if self.pin_memory:
            if image_type == 'masked_image':
                
                self.masked_image_list = data_loader_parallel.load_masked_image_tensor_parallel()
                
            elif image_type == 'masked_origin_image':
                self.masked_origin_image_list = data_loader_parallel.load_tensor_parallel(image_type)
                self.masked_origin_image_list = np.expand_dims(self.masked_origin_image_list, axis=1)
            
            else:
                
                self.image_list = data_loader_parallel.load_image_tensor_parallel()
                
        self.image_mean = np.load(os.path.join(stats_dir, image_type + '_mean.npy'))
        self.image_std  = np.load(os.path.join(stats_dir, image_type + '_std.npy'))

        self.velocity_list = data_loader_parallel.load_velocity_tensor_parallel()
        self.label_list = data_loader_parallel.load_label_tensor_parallel()
        
        
        # Only for confusion
        if num_debug_samples:
            self.indices = self.indices[:num_debug_samples]

    def __len__(self):
        # Returns the length of the dataset. Defaults to 'image_wise' split.
        return len(self.indices)

    def __getitem__(self, idx):
        
        idx   = self.indices[idx]

        # Save data in pinned memory
        if self.pin_memory:
            if self.image_type == 'masked_image':
                image = self.masked_image_list[idx]
                image = (image - self.image_mean) / self.image_std

            elif self.image_type == 'masked_origin_image':
                image = self.masked_origin_image_list[idx]
                image = (image - self.image_mean) / self.image_std

            else: 
                image = self.image_list[idx]
                image = (image - self.image_mean) / self.image_std
                
        # Not save data in pinned memory. Use only you dont have enough memory.
        else:
            tensor_name = ("%s_%0" + str(self.FILE_ZERO_PADDING_NUM) + "d.npy")%(self.image_type, idx)
            image = np.load(os.path.join(self.tensor_dir, tensor_name), allow_pickle=True).astype(np.float32)
            image = np.expand_dims(image, axis=0)
            image = (image - self.image_mean) / self.image_std

        velocity = self.velocity_list[idx]
        velocity = (velocity - self.velocity_mean) / self.velocity_std
        
        label = self.label_list[idx]
        label_onehot = torch.from_numpy(np.eye(2)[int(label)].astype(np.float32)) # one-hot encoding
        
        return image, velocity, label_onehot


def main():
    # Get current file path
    current_file_path = os.path.dirname(os.path.realpath(__file__))
    config_file = os.path.abspath(os.path.join(current_file_path, '..', '..',  'config', 'config.yaml'))
    
    # Load configuation file
    with open(config_file,'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    FILE_ZERO_PADDING_NUM = cfg['file_zero_padding_num']
    
    dexnet_train_dataset = PushNetDataset(cfg["data_path"], type='train', zero_padding=FILE_ZERO_PADDING_NUM)
    train_dataloader = DataLoader(dexnet_train_dataset, batch_size=64, shuffle=True, num_workers=16)
        
    # test data loader
    for i, (image, velocity, label) in enumerate(train_dataloader):

        print(len(train_dataloader))
        print('image.shape:', image.shape)
        print('velocity.shape:', velocity.shape)
        print('label.shape:', label.shape)
        print('image.device:', image.device)
        print('velocity.device:', velocity.device)
        print('label.device:', label.device)
        
        image = image.permute(0, 2, 3, 1)
        image_tf = image.reshape(-1, image.shape[2], image.shape[3])
        
        img_mean = torch.mean(image_tf)
        
        plt.imshow(image_tf)
        plt.show()
        break
    
if __name__ == '__main__':
    main()
