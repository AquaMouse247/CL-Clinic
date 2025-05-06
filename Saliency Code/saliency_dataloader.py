import torch
from torch.utils.data import Dataset, Subset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import os
import cv2
import copy, math, random


class SalDataloader:
    def __init__(self, args):
        self.args = args
        self.algorithm = args.algorithm
        self.dataset = args.dataset
    
    def get_indices(self, dataset, class_name):
        indices = []
        for i in range(len(dataset.targets)):
            for j in class_name:
                if dataset.targets[i] == j:
                    indices.append(i)
        return indices
    
    def load_data(self, desired_classes, imgs_per_class):
        transform = transforms.Compose(
            [transforms.ToTensor()])
    
        if not os.path.isdir(f"SaliencyMaps/{self.algorithm}/" + self.dataset):
            os.makedirs(f"SaliencyMaps/{self.algorithm}/" + self.dataset)
    
        saliency_set = torch.utils.data.Dataset()
        mean = None
        std = None
    
        if self.dataset == "mnist":
            if self.algorithm == "foster" or self.algorithm == "memo" or self.algorithm == "der":
                trsf = transforms.Compose([transforms.Pad(2), transforms.ToTensor()])
            elif self.algorithm == "DGR":
                trsf = transforms.Compose([transforms.ToTensor(), transforms.ToPILImage(),
                                           transforms.Pad(2), transforms.ToTensor()])
            else:
                trsf = transforms.Compose([transforms.ToTensor()])
            saliency_set = datasets.MNIST(root=f"Datasets/{self.dataset}/", train=False, download=True, transform=trsf)
        elif self.dataset == "svhn":
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
    
            saliency_set = datasets.SVHN(root=f"Datasets/{self.dataset}/", split='test',
                                        download=True,
                                        transform=transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean, std)]))
            saliency_set.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            saliency_set.targets = saliency_set.labels
    
        elif self.dataset == "cifar10":
            mean = torch.tensor([0.4914, 0.4822, 0.4465])
            std = torch.tensor([0.2023, 0.1994, 0.2010])
    
            saliency_set = datasets.CIFAR10(root=f"Datasets/{self.dataset}/", train=False,
                                           download=True,
                                           transform=transforms.Compose(
                                               [transforms.ToTensor(),
                                                transforms.Normalize(mean, std)]))
    
        elif self.dataset == "cifar100":
            mean = torch.tensor([0.5071, 0.4867, 0.4408])
            std = torch.tensor([0.2675, 0.2565, 0.2761])
    
            saliency_set = datasets.CIFAR100(root=f"Datasets/{self.dataset}/", train=False,
                                            download=True,
                                            transform=transforms.Compose(
                                                [transforms.ToTensor(),
                                                 transforms.Normalize(mean, std)]))
    
    
        idx = self.get_indices(saliency_set, desired_classes)
    
        subset = Subset(saliency_set, idx)
        #if self.algorithm == "DGR":
        #    self.formatDatasetForDGR(subset)

        # Create a DataLoader for the subset
        saliencyLoader = DataLoader(subset, batch_size=256)

        if imgs_per_class is None:
            return saliencyLoader

        images, labels = next(iter(saliencyLoader))
    
        # Order saliency images and labels by class
        sal_idx = []
        sal_labels = []
        for i in range(len(desired_classes)):
            num = 0
            while len(sal_idx) < imgs_per_class * (i + 1):
                if labels[num] == desired_classes[i]:
                    sal_idx.append(num)
                    sal_labels.append(desired_classes[i])
                num += 4
                #num += 16
                #num += 1
        sal_imgs = images[sal_idx]
    
        return sal_imgs, torch.tensor(sal_labels), saliency_set.classes, mean, std
    
    def load_validation_data(self):
        import cv2
        import torch
        from torchvision.utils import make_grid
    
        labels = [2, 2, 2, 2, 2, 5, 5, 5, 5, 5]
        classes = ('0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
                   '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine')
    
        images = []
        # Square
        # square = 255/2 * np.ones(shape=(28, 28, 1), dtype=np.float32)
        square = np.ones(shape=(28, 28, 1), dtype=np.float32)
        cv2.rectangle(square,
                      pt1=(0, 0),
                      pt2=(27, 27),
                      color=(255, 255, 255),
                      thickness=4)
        images.append(square)
        images.append(square)
    
        # X
        X = np.zeros(shape=(28, 28, 1), dtype=np.float32)
        cv2.line(X,
                 pt1=(0, 0),
                 pt2=(27, 27),
                 color=(255, 255, 255),
                 thickness=4)
        cv2.line(X,
                 pt1=(27, 0),
                 pt2=(0, 27),
                 color=(255, 255, 255),
                 thickness=4)
        images.append(X)
    
        # L
        L = np.zeros(shape=(28, 28, 1), dtype=np.float32)
        cv2.rectangle(L,
                      pt1=(0, 0),
                      pt2=(0, 27),
                      color=(255, 255, 255),
                      thickness=1)
        cv2.rectangle(L,
                      pt1=(0, 27),
                      pt2=(27, 27),
                      color=(255, 255, 255),
                      thickness=1)
        images.append(L)
    
        # Reverse L
        LReverse = np.zeros(shape=(28, 28, 1), dtype=np.float32)
        cv2.rectangle(LReverse,
                      pt1=(0, 27),
                      pt2=(27, 27),
                      color=(255, 255, 255),
                      thickness=1)
        cv2.rectangle(LReverse,
                      pt1=(27, 27),
                      pt2=(27, 0),
                      color=(255, 255, 255),
                      thickness=1)
        images.append(LReverse)
    
        # Corners
        corners = np.zeros(shape=(28, 28, 1), dtype=np.float32)
        cv2.rectangle(corners, pt1=(0, 0), pt2=(1, 1), color=(255, 255, 255), thickness=1)
        cv2.rectangle(corners, pt1=(0, 27), pt2=(1, 26), color=(255, 255, 255), thickness=1)
        cv2.rectangle(corners, pt1=(27, 0), pt2=(26, 1), color=(255, 255, 255), thickness=1)
        cv2.rectangle(corners, pt1=(27, 27), pt2=(26, 26), color=(255, 255, 255), thickness=1)
        images.append(corners)
    
        # Top
        top = np.zeros(shape=(28, 28, 1), dtype=np.float32)
        cv2.line(top,
                 pt1=(0, 0),
                 pt2=(27, 0),
                 color=(255, 255, 255),
                 thickness=1)
        images.append(top)
    
        # Left
        left = np.zeros(shape=(28, 28, 1), dtype=np.float32)
        cv2.line(left,
                 pt1=(0, 0),
                 pt2=(0, 27),
                 color=(255, 255, 255),
                 thickness=4)
        images.append(left)
    
        # Bottom
        bottom = np.zeros(shape=(28, 28, 1), dtype=np.float32)
        cv2.line(bottom,
                 pt1=(0, 27),
                 pt2=(27, 27),
                 color=(255, 255, 255),
                 thickness=4)
        images.append(bottom)
    
        # Right
        right = np.zeros(shape=(28, 28, 1), dtype=np.float32)
        cv2.line(right,
                 pt1=(27, 0),
                 pt2=(27, 27),
                 color=(255, 255, 250),
                 thickness=1)
        images.append(right)
    
        images = np.array(images)
        imgs_tensor = torch.from_numpy(images).permute(0, 3, 1, 2)
        grid = make_grid(imgs_tensor)
        grid_np = grid.permute(1, 2, 0).numpy()
        # plt.imshow(grid_np, cmap="gray")
        # plt.show()
    
        return imgs_tensor, torch.tensor(labels), classes, None, None
    
    def formatDatasetForDGR(self, subset):
        capacity = 32 * 3000
        if capacity is not None:
            if len(subset) < capacity:
                # Concatenate multiple copies of the subset to reach the desired capacity
                subset = ConcatDataset([copy.deepcopy(subset) for _ in range(math.ceil(capacity / len(subset)))])
            else:
                # If the subset already exceeds the capacity, randomly sample from it
                indices = random.sample(range(len(subset)), capacity)
                subset = Subset(subset, indices)