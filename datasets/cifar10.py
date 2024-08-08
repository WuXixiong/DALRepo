from torchvision import datasets
from torch.utils.data import Dataset, Subset
import numpy as np
from arguments import parser  # Assuming parser is defined in the arguments module

class MyCIFAR10(Dataset):
    def __init__(self, file_path, train, download, transform):
        self.cifar10 = datasets.CIFAR10(file_path, train=train, download=download, transform=transform)
        self.targets = np.array(self.cifar10.targets)
        self.classes = self.cifar10.classes

        args = parser.parse_args()
        
        if args.imbalanceset:
            # Create imbalance ratios
            imbalance_ratios = np.linspace(args.imb_ratio, 1, 10)

            # Get index of each class
            train_targets = np.array(self.cifar10.targets)
            train_idx_per_class = [np.where(train_targets == i)[0] for i in range(10)]

            # Resample according to the indices
            new_indices = []
            for class_idx, class_indices in enumerate(train_idx_per_class):
                n_samples = int(len(class_indices) * imbalance_ratios[class_idx])
                new_indices.extend(np.random.choice(class_indices, n_samples, replace=False))

            # Create the imbalanced train dataset
            self.cifar10.data = self.cifar10.data[new_indices]
            self.targets = self.targets[new_indices]

            if args.imb_type in ["same", "different"]:
                # Create test set imbalance ratios
                if args.imb_type == "different":
                    imbalance_ratios = imbalance_ratios[::-1]  # Reverse for different imbalance

                test_set = datasets.CIFAR10(file_path, train=False, download=download, transform=transform)
                test_targets = np.array(test_set.targets)
                test_idx_per_class = [np.where(test_targets == i)[0] for i in range(10)]

                # Resample test set
                new_test_indices = []
                for class_idx, class_indices in enumerate(test_idx_per_class):
                    n_samples = int(len(class_indices) * imbalance_ratios[class_idx])
                    new_test_indices.extend(np.random.choice(class_indices, n_samples, replace=False))

                self.test_set = Subset(test_set, new_test_indices)

    def __getitem__(self, index):
        data, _ = self.cifar10[index]
        target = self.targets[index]
        return data, target, index

    def __len__(self):
        return len(self.cifar10)