from torch.utils.data import Dataset
import random
from data_utils import get_hard_neg_dict

class HardNegContrastiveDataset(Dataset):
    def __init__(self, model, loader, dataset, dataset_transform, hard_neg_transform):
        self.dataset = dataset
        self.dataset_transform = dataset_transform
        self.hard_neg_transform = hard_neg_transform
        self.hard_neg_dict = get_hard_neg_dict(model, loader)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = self.dataset_transform(img)

        hard_negs = self.hard_neg_dict.get(label, [])
        if hard_negs:
            hard_neg, hard_neg_label = random.choice(hard_negs)
            hard_neg = self.hard_neg_transform(hard_neg)
        else:
            while True:
                neg_idx = random.choice(range(len(self.dataset)))
                if self.dataset[neg_idx][1] != label:
                    break
            hard_neg, hard_neg_label = self.dataset[neg_idx]
            hard_neg = self.hard_neg_transform(hard_neg)
            hard_neg_label = hard_neg_label.item()

        return img, label, hard_neg, hard_neg_label

