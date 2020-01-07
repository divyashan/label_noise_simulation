import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import yaml
import os


def make_hdf5_file(model, tta_dataset, device, config):
    file_dataset = config["data_loader"]["hdf5"]
    n_classes  = config["test"]["num_classes"]
    f = h5py.File("datasets/{}".format(file_dataset), "w")

    augs = list(tta_dataset[0].keys())
    augs.remove("label")
   
    for name in augs:
        if type(tta_dataset[0][name])== list:
            length = len(tta_dataset[0][name])
        else:
            length = 1
        dataset = f.create_dataset("{}_output".format(name), (len(tta_dataset), length, n_classes)) 
    label = f.create_dataset("label", (len(tta_dataset), 1))

    tta_dataset = torch.utils.data.DataLoader(tta_dataset, batch_size=1,
                                         shuffle=False, num_workers=1)

    model.eval()
    for i, data in enumerate(tta_dataset):
        print("Loading sample {}".format(i))
        f["label"][i] = data["label"]
        for aug_name in augs:
            if aug_name != "five_crop" and not aug_name.startswith("rot_"):
                input_var = data[aug_name].float().to(device)
                output = model(input_var)
                output = output.detach().cpu()
                f["{}_output".format(aug_name)][i, 0, :]  = output
            else:
                for j in range(len(data[aug_name])):
                    input_var = data[aug_name][j].float().to(device)
                    output = model(input_var)
                    output = output.detach().cpu()
                    f["{}_output".format(aug_name)][i, j, :] = output

class TTA_HDF5_Dataset(Dataset):

    def __init__(self, model, tta_dataset, device, config_file):
        if type(config_file) == dict:
            config = config_file
        else: 
            with open(config_file) as fp:
                config = yaml.load(fp)
        file_dataset = config["data_loader"]["hdf5"]
        if not os.path.isfile("datasets/{}".format(file_dataset)):
            make_hdf5_file(model, tta_dataset, device, config)
        f = h5py.File("datasets/{}".format(file_dataset))

        self.data = {}
        for key in f.keys():
            self.data[key] = f.get(key)
        self.length = len(tta_dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {key: self.data[key][idx] for key in self.data} 
