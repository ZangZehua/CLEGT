from torchvision.transforms import transforms

from torchvision import transforms, datasets
from exceptions.exceptions import InvalidDatasetSelection
from data_aug.transform import Transform


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    def get_dataset(self, name):
        valid_datasets = {
            'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True, transform=Transform(32), download=True),
            'cifar100': lambda: datasets.CIFAR100(self.root_folder,train=True,transform=Transform(32),download=True),
            'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled', transform=Transform(96), download=True)
        }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
