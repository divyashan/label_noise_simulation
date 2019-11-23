import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class TTADataset(Dataset):
    def __init__(self, original_dataset, augmentations):
        self.original_dataset = original_dataset
        self.augmentations = augmentations 
    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        sample = {"image": self.original_dataset[idx][0],
                    "label": self.original_dataset[idx][1]}

        to_pil_image = transforms.ToPILImage()
        to_tensor = transforms.ToTensor() 
        if "hflip" in self.augmentations:
            hflip = to_tensor(transforms.functional.hflip(to_pil_image(self.original_dataset[idx][0])))
            sample["hflip"] = hflip
        if "five_crop" in self.augmentations:
            c, h, w = self.original_dataset[idx][0].shape
            size = int(self.original_dataset[idx][0].shape[1] * 0.875)
            five_crop_transform = transforms.Compose([transforms.FiveCrop(size), # this is a list of PIL Images
                                           transforms.Lambda(lambda crops: [transforms.ToTensor()(transforms.Resize((h, w))(crop)) for crop in crops]) # returns a 4D tensor
                                          ])
            five_crop = five_crop_transform(to_pil_image(self.original_dataset[idx][0]))
            sample["five_crop"] = five_crop
        return sample
             

