from torch.utils.data import Dataset
from simulation import gen_corrupted_labels
import random
import torch

def add_noise_to_labels(trainset, delta_matrix):
   tilde_labels = gen_corrupted_labels(delta_matrix, trainset.targets) 
   return tilde_labels

class NoisyDataset(Dataset):
    """
    Stores a subset of the Drive Ahead dataset for which a data frame has
    already been created.
    Parameters:
        frame: a csv file that maps image file names to quaternion poses
        dataset_path: the directory where the images are stored
        photo_type:  "ir/" or "depth/" depending on whether IR or depth images
            are of interest
    """

    def __init__(self, original_dataset, delta_matrix):
        self.original_dataset = original_dataset
        self.delta_matrix = delta_matrix
        self.corrupted_labels = add_noise_to_labels(original_dataset, delta_matrix)
        corrupted_label_counts = {i: list(self.corrupted_labels).count(i) for i in set(self.corrupted_labels)}
        class_weights = [0 for i in range(len(corrupted_label_counts.keys()))]
        max_class_size = max(corrupted_label_counts.values())
        for i in range(len(class_weights)):
            class_weights[i] += corrupted_label_counts[i]/max_class_size
        self.class_weights = torch.tensor(class_weights)
    
    def __len__(self):
        return len(self.corrupted_labels)

    def __getitem__(self, idx):
        return self.original_dataset[idx][0], self.corrupted_labels[idx]

