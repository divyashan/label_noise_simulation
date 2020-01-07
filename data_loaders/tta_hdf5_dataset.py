import torch
from torch.utils.data import Dataset
import h5py
import yaml
import os


def make_hdf5_file(model, tta_dataset, device, config):
    file_dataset = config["hdf5"]
    n_classes  = config["test"]["n_classes"]
    f = h5py.File("datasets/{}".format(file_dataset), "w")
    image_output = f.create_dataset("image_output", (len(tta_dataset), 1, n_classes))
    five_crop_output = f.create_dataset("five_crop_output", (len(tta_dataset), 5, n_classes))
    hflip_output = f.create_dataset("hflip_output", (len(tta_dataset), 1, n_classes))
    rot_output = f.create_dataset("rot_output", (len(tta_dataset), tta_dataset.num_rotations, n_classes)) 
    label = f.create_dataset("label", (len(tta_dataset), 1))

    augs = tta_dataset[i].get_keys()
    augs.remove("label")
    for i in range(len(tta_dataset)):
        f["label"][i] = tta_dataset[i]["label"]
        for aug_name in imgs:
            if aug_name != "five_crop" and aug_name != "hflip":
                input_var = tta_dataset[aug_name].float().to(device)
                output = model(input_var)
                output = output.detach().cpu()
                f["{}_output".format(aug_name)][i, 0, :]  = output
            else:
                for j in range(len(tta_dataset[aug_name])):
                    input_var = tta_dataset[aug_name][j].float().to(device)
                    output = model(input_var)
                    output = output.detach().cpu()
                    f["{}_output".format(aug_name)][i, j, :] = output

class TTA_HDF5_Dataset(Dataset):

    def __init__(self, model, tta_dataset, device, config):
        if type(config_file) == dict:
            config = config_file
        else: 
            with open(config_file) as fp:
                config = yaml.load(fp)
        file_dataset = config["hdf5"]
        if not os.path.isfile("datasets/{}".format(file_dataset)):
            make_hdf5_file(model, tta_dataset, device)
        f = h5py.File("datasets/{}".format(file_dataset)

        self.label = f.get('label')
        self.image_output = f.get('image_output')
        self.five_crop_output = f.get('five_crop_output')
        self.hflip_output = f.get('hflip_output')
        self.rot_output = f.get('rot_output')

        self.length = len(tta_dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {"label": self.label[i], "image_output": self.image_output[i],
                "five_crop_output": self.five_crop_output[i], "hflip_output": self.hflip_output[i],
                "rot_output": self.rot_output[i]}

