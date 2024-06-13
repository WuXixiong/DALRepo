from torchvision import datasets
from torch.utils.data.dataset import Dataset
import numpy as np

from arguments import parser

class MyCIFAR10(Dataset):
    def __init__(self, file_path, train, download, transform):
        self.cifar10 = datasets.CIFAR10(file_path, train=train, download=download, transform=transform)
        # 获得原始目标数组
        self.targets = np.array(self.cifar10.targets)
        self.classes = self.cifar10.classes

            # 定义每个类别的目标样本数
        args = parser.parse_args()
        if args.imbalanceset:
            args.target_sample_count = [500, 1000, 1500, 2000, 500, 300, 700, 800, 900, 400]  # 每个数字代表该类别希望保留的样本数
            # 创建新的数据和目标数组
            indices = []
            for i in range(10):  # CIFAR10有10个类别
                # 找到属于该类的所有索引
                class_indices = np.where(self.targets == i)[0]
                # 如果定义的样本数多于实际样本数，则进行重复抽样，否则进行随机抽样
                if len(class_indices) > args.target_sample_count[i]:
                    chosen_indices = np.random.choice(class_indices, args.target_sample_count[i], replace=False)
                else:
                    chosen_indices = np.random.choice(class_indices, args.target_sample_count[i], replace=True)
                indices.extend(chosen_indices)
        
            # 重置 cifar10 以只包含选择的索引
            self.cifar10.data = self.cifar10.data[indices]
            self.targets = self.targets[indices]

    def __getitem__(self, index):
        # 获取数据和目标，因为数据集已经是调整后的，直接返回即可
        data, _ = self.cifar10[index]
        target = self.targets[index]
        return data, target, index

    def __len__(self):
        # 数据集长度为调整后的长度
        return len(self.cifar10)
