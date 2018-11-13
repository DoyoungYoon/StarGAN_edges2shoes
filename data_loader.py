from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import numpy as np
import random



class Edge2shoes(data.Dataset):

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        c=0
        list = [0, 0, 0, 0, 0]      #각 라벨에 해당하는 데이터 개수

        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[0].split(',')
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[1:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split(',')
            filename = split[0].replace('-','.')
            filename = '{0:0.6f}'.format(float(filename))
            values = split[1:]                                                          #각 파일들의 values 전체 label들의 값
            label = []
            for attr_name in self.selected_attrs:                                   #내가 선택한 label들을 하나씩 불러들어옴
                idx = self.attr2idx[attr_name] - 1                                  #걔네가 몇번째에 있는지 index
                #print(values[idx] )
                label.append(values[idx] == '1')                                            #그 label중 1인 건 True, 0 인건 False로 붙여줌
            #label.append(False)
            if not True in label:
                c
            else:
                count = 0
                for label_in_true in label:
                    if label_in_true == True:
                        count = count+1
                if count == 1:
                    c = c+1
                    for i, a in enumerate(label):
                        if a == True:
                            list[i] +=1
                    #print(label)
                    if(c+1)<2000:
                        self.test_dataset.append([filename, label])
                    else:
                        self.train_dataset.append([filename, label])
            #print(label)
        print(list)
        print('data : '+str(c))

        print('Finished preprocessing the edges2shoes dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename+'.jpg'))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        return self.num_images


def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset='edges2shoes', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == 'edges2shoes':
        dataset = Edge2shoes(image_dir, attr_path, selected_attrs, transform, mode)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)


    return data_loader