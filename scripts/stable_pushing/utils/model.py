#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from .cosine_normalization import CosineNormalization

class PushNet(nn.Module):
    def __init__(self):
        
        super(PushNet, self).__init__()
        
        self.im_stream = nn.Sequential(
            # conv1_1 
            nn.Conv2d(1, 16, 9, stride=1, padding="valid"),  # 9x9x1 16 filters
            # nn.ReLU(),
            nn.LeakyReLU(),
            # conv1_2
            nn.Conv2d(16, 16, 5, stride=1, padding="valid"), # 5x5x16 16 filters
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2),
            # conv2_1
            nn.Conv2d(16, 16, 5, stride=1, padding="valid"),  # 5x5x16 16 filters
            # nn.ReLU(),
            nn.LeakyReLU(),
            
            # conv2_2
            nn.Conv2d(16, 16, 5, stride=1, padding="valid"),  # 5x5x16 16 filters #(B, C, H , W)
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2),
            # fc3
            nn.Flatten(1,3), # (B, C*H*W)
            nn.Linear(17*17*16,128),
            # nn.ReLU(),
            nn.LeakyReLU(),
        )
        self.velocity_stream = nn.Sequential(
            # pc1
            nn.Linear(3, 16),
            nn.LeakyReLU(),
            # nn.ReLU()
        )
        self.merge_stream = nn.Sequential(
            # fc4
            nn.Linear(128+16, 128), # (B, ..., C)
            nn.LeakyReLU(),
            # nn.ReLU(),
            # fc5
            CosineNormalization(128, 2),
        )
        
    def forward(self, image, velocity):
        image = self.im_stream(image)
        velocity = self.velocity_stream(velocity)
        feature = torch.cat((image, velocity), 1)
        return self.merge_stream(feature)

    def calculate_output_size(input_size, kernel_size, stride, padding):
        return int((input_size - kernel_size + 2*padding)/stride) + 1
    
if __name__ == '__main__':
    device = torch.device('cpu')
    if torch.cuda.is_available():
        print("CUDA is available")
        device = torch.device('cuda')

    model = PushNet().eval()
    print(list(model.children()))

# NewModel
class NewModel(nn.Module):
    def __init__(self, original_model, *args):
        super().__init__()
        
        self.im_stream = nn.Sequential(*list(original_model.children())[0])
        self.pose_stream = nn.Sequential(*list(original_model.children())[1])
        self.merge_stream = nn.Sequential(*list(original_model.children())[2][:2])

    def forward(self, image, pose):
        image = self.im_stream(image)
        pose = self.pose_stream(pose)
        feature = torch.cat((image, pose), 1)
        return self.merge_stream(feature)