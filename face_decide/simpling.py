from torch.utils.data import Dataset,DataLoader
import os
import numpy as np
import torch
from PIL import Image

class FaceDataset(Dataset):

    def __init__(self, path):
        self.path = path
        self.dataset = []
        self.dataset.extend(open(os.path.join(path, "positive.txt")).readlines())
        # print(self.dataset)
        self.dataset.extend(open(os.path.join(path, "negative.txt")).readlines())
        self.dataset.extend(open(os.path.join(path, "part.txt")).readlines())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        strs = self.dataset[index].strip().split(" ")
        for i, a in enumerate(strs):
            strs[i] = a.strip("[],")
        img_path = os.path.join(self.path, strs[0])
        cls = torch.tensor([int(strs[1])],dtype=torch.float32)
        offset_datas = strs[2:]
        offset = [float(offset_data) for offset_data in offset_datas]
        offset = torch.tensor(offset)

        img_data = torch.tensor(np.array(Image.open(img_path)) / 255. - 0.5,dtype=torch.float32)
        # print(img_data.shape)
        img_data = img_data.permute(2,0,1)

        return img_data, cls, offset



if __name__ == '__main__':
    dataset = FaceDataset(r"D:\data\wflw_data")
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    print(dataset[0][2].shape)
    '''
    
    
    
    
    
    dataset：加载的数据集(Dataset对象)
    batch_size：batch size
    shuffle:：是否将数据打乱
    num_workers：使用多进程加载的进程数，0代表不使用多进程
    '''
    dataloader = DataLoader(dataset,5,shuffle=True,num_workers=4)
    for i ,(img,cls,offset) in enumerate(dataloader):
        print(img.shape)
        print(cls.shape)
        print(cls)
        print(offset.shape)
