import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class TTADataset(Dataset):
    def __init__(self, original_dataset, augmentations):
        self.original_dataset = original_dataset
        self.augmentations = augmentations 
        self.num_rotations = 0
        for aug in self.augmentations:
            if aug.startswith("rot_"):
                self.num_rotations = int(aug.split("_")[1])
       
    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        sample = {"image": self.original_dataset[idx][0],
                    "label": self.original_dataset[idx][1]}

        to_pil_image = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()

        pil_version = to_pil_image(self.original_dataset[idx][0])
        if "hflip" in self.augmentations:
            hflip = to_tensor(transforms.functional.hflip(pil_version))
            sample["hflip"] = hflip
        if "five_crop" in self.augmentations:
            c, h, w = self.original_dataset[idx][0].shape
            size = int(self.original_dataset[idx][0].shape[1] * 0.875)
            five_crop_transform = transforms.Compose([transforms.FiveCrop(size), # this is a list of PIL Images
                                           transforms.Lambda(lambda crops: [transforms.ToTensor()(transforms.Resize((h, w))(crop)) for crop in crops]) # returns a 4D tensor
                                          ])
            five_crop = five_crop_transform(pil_version)
            sample["five_crop"] = five_crop
        if self.num_rotations != 0:
            rotation_transforms = transforms.Compose([transforms.RandomRotation(degrees=15), transforms.ToTensor()])
            rotation = [rotation_transforms(pil_version) for i in range(self.num_rotations)]
            sample["rot_{}".format(self.num_rotations)]= rotation
        return sample
             

