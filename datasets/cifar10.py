from torchvision import datasets
from torch.utils.data.dataset import Dataset
import numpy as np

from arguments import parser

class MyCIFAR10(Dataset):
    def __init__(self, file_path, train, download, transform):
        self.cifar10 = datasets.CIFAR10(file_path, train=train, download=download, transform=transform)
        self.targets = np.array(self.cifar10.targets)
        self.classes = self.cifar10.classes

        args = parser.parse_args()
        if args.imbalanceset:
            args.target_sample_count = [500, 1000, 1500, 2000, 500, 300, 700, 800, 900, 400]  # Set the defined imbalance classes
            indices = []
            for i in range(10):  # Ten classes in CIFAR10
                # Get all index belong to this class
                class_indices = np.where(self.targets == i)[0]
                # If defined samples larger than actually samples, do resampling
                if len(class_indices) > args.target_sample_count[i]:
                    chosen_indices = np.random.choice(class_indices, args.target_sample_count[i], replace=False)
                else: # Otherwise, randomly sampling
                    chosen_indices = np.random.choice(class_indices, args.target_sample_count[i], replace=True)
                indices.extend(chosen_indices)
        
            # Reset dataset
            self.cifar10.data = self.cifar10.data[indices]
            self.targets = self.targets[indices]

    def __getitem__(self, index):
        data, _ = self.cifar10[index]
        target = self.targets[index]
        return data, target, index

    def __len__(self):
        return len(self.cifar10)
