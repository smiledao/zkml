import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


class ExpressionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super(ExpressionDataset, self).__init__()
        self.transform = transform
        self.samples = []
        self.labels = []

        label_counter = 0
        for dirpath, _, filenames in os.walk(root_dir):
            label = os.path.split(dirpath)[-1]
            if label == "happy":
                label = 1
            else:
                label = 0
            for filename in filenames:
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    filepath = os.path.join(dirpath, filename)
                    image = Image.open(filepath).convert('RGB')
                    if self.transform is not None:
                        image = self.transform(image)
                    self.samples.append(image)
                    self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]
