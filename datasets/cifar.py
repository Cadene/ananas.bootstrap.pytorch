import torch
import torchvision
from torchvision import transforms
from bootstrap.datasets.dataset import Dataset
from tqdm import tqdm
from pprint import pprint

class CIFAR(Dataset):

    def __init__(self, dir_data,
                 split='train',
                 batch_size=4,
                 shuffle=False,
                 pin_memory=False,
                 nb_threads=4,
                 name='CIFAR10',
                 im_tf=None):
        super(CIFAR, self).__init__(
            dir_data,
            split,
            batch_size,
            shuffle,
            pin_memory,
            nb_threads)
        self.name = name

        if im_tf is None:
            mean = [0.49139968, 0.48215827, 0.44653124]
            std = [0.24703233, 0.24348505, 0.26158768]
            self.im_tf = {}
            self.im_tf['train'] = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            self.im_tf['val'] = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            self.im_tf = im_tf

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck')
        self.dataset = torchvision.datasets.__dict__[self.name](
            dir_data,
            train=split=='train',
            download=True,
            transform=self.tf[split])

    def __getitem__(self, index):
        img, target = self.dataset[index]
        item = {}
        item['index'] = index
        item['data'] = img
        item['class_id'] = torch.LongTensor([target])
        return item

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    dataset = CIFAR('data/cifar10', name='CIFAR10')
    pprint(dataset[0])
    print(dataset[0]['data'].shape)
    for item in tqdm(dataset):
        pass


    dataset = CIFAR('data/cifar100', name='CIFAR100')
    pprint(dataset[0])
    print(dataset[0]['data'].shape)
    for item in tqdm(dataset):
        pass

