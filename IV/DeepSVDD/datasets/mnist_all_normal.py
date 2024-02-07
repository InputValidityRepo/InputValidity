from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import MNIST
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import global_contrast_normalization
import torchvision.transforms as transforms
import imageio
import torch

class MNIST_Dataset_all_normal(TorchvisionDataset):

    def __init__(self, root: str):
        super().__init__(root)
        self.n_classes = 2  # 0: normal, 1: outlier
        # Pre-computed min and max values (after applying GCN) from train data per class
        min_max_overall = [(-0.8826567065619495, 20.108062262467364)]

        # MNIST preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([min_max_overall[0][0]],
                                                             [min_max_overall[0][1] - min_max_overall[0][0]])])

        #target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))
        target_transform = None
        self.train_set = MyMNIST(root=root, train=True, download=False,
                                transform=transform, target_transform=target_transform)
    def __len__(self):
        return len(self.train_set)

class MyMNIST(MNIST):
    """Torchvision MNIST class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, *args, **kwargs):
        super(MyMNIST, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """Override the original method of the MNIST class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            print('enter test')
            img, target = self.test_data[index], self.test_labels[index]
        imageio.imwrite(
            'original_MNIST' + '.png',
            img)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        # save untransformed data

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor([0]), torch.tensor([0])  # only line changed

