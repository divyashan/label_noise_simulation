import argparse
import torch
import yaml
import os
from modules import get_model
import math
import torchvision
import torchvision.transforms as transforms
from tta_dataset import TTADataset
from tensorboardX import SummaryWriter

DEFAULT_CONFIG = os.path.dirname(__file__) + "configs/mnist_test.yaml"


def get_tta_dataset(dataset, augmentations, batch_size):
    cifar10_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    mnist_transform = transforms.Compose([transforms.ToTensor()]) 

    if dataset =="CIFAR10":
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=cifar10_transform)
    elif dataset == "MNIST":
        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=mnist_transform)
    testset = TTADataset(testset, augmentations)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=1)

    return testloader


def load_model(model_path, device, config):
    model_name = config["test"]["model"]
    num_classes = config["test"]["num_classes"]
    num_channels = config["test"]["num_channels"]
    model = get_model(model_name, pretrained=False, num_channels=num_channels, num_classes=num_classes)
    model.to(device)
    print("Loading {} into {}".format(model_path, model_name))
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["state_dict"])
    return model, num_classes

def main():
    """ Loads arguments and starts training."""
    parser = argparse.ArgumentParser(description="Simulated Noise Experiments")
    parser.add_argument('-c', '--config', default=DEFAULT_CONFIG, type=str)

    args = parser.parse_args()
    config_file = args.config
    # Load config
    assert os.path.exists(args.config), "Config file {} does not exist".format(
        args.config)

    with open(config_file) as fp:
        config = yaml.load(fp, yaml.Loader)

    device = torch.device(config["test"]["device"] if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    models_names = config["test"]["models_names"]
    models_paths = config["test"]["models_paths"]
    true_delta_matrices = config["test"]["true_delta_matrices"]
    batch_size = 1
    augmentations = config["data_loader"]["augmentations"]
    print("TTA AUGMENTATIONS: {}".format(augmentations))
    val_loader = get_tta_dataset(config["data_loader"]["name"], augmentations, batch_size)
    writer_val = SummaryWriter(
        "runs/evaluate/validation")


    for i in range(len(models_names)):
        model, num_classes = load_model(models_paths[i], device, config)
        run_evaluation(models_names[i], torch.Tensor(true_delta_matrices[i]), augmentations, model, device, num_classes, val_loader, writer_val)
        
def run_evaluation(name_model, true_delta_matrix, augmentations, model, device, num_classes, val_loader, writer):
        mse = torch.nn.MSELoss(reduction = "sum")
        model.eval()
        eval_fraction = 1
        val_load_iter = iter(val_loader)
        counts = torch.zeros(num_classes)
        label_noise_matrix = torch.zeros(num_classes, num_classes)
        size = int(len(val_loader) * eval_fraction)
        accuracy = 0
        for i in range(size):
            data = val_load_iter.next()
            target_var = data["label"].float().to(device)
            output, pred_class = run_inference_on_augmentations(model, data, augmentations, num_classes, device, writer)
            if pred_class.item() == target_var.item():
                accuracy += 1
                # compute output
            label_noise_matrix[int(target_var.item())] += output 
            counts[int(target_var.item())] += 1.
           # measure accuracy and record loss
        for j in range(num_classes):
            label_noise_matrix[j] /= counts[j]

        error = mse(label_noise_matrix, true_delta_matrix)
        print("Model: {}, LM Error: {}, Acc: {}".format(name_model, error, accuracy/size))

def run_inference_on_augmentations(model, data, augmentations, num_classes, device, writer):
    aggregated = torch.zeros(num_classes).float()
    num_total_augs = 0 
    softmax = torch.nn.Softmax(dim=1)
    for aug in augmentations:
        if  type(data[aug]) == list:
            for i in range(len(data[aug])):
               example = data[aug][i].float().to(device)
               writer.add_image("five_crop", data[aug][i][0], i) 
               output = softmax(model(example)).detach().reshape(num_classes).cpu()
               aggregated += output 
            num_total_augs += len(data[aug]) 
        else:
            example = data[aug].float().to(device)
            output = softmax(model(example)).detach().reshape(num_classes).cpu()
            aggregated += output
            num_total_augs += 1
            writer.add_image("hflip", data[aug][0], 0) 
   
    aggregated += softmax(model(data["image"].float().to(device))).detach().reshape(num_classes).cpu()
    writer.add_image("original", data["image"][0], 0) 

    final_output = aggregated/(num_total_augs + 1)
    return final_output, torch.argmax(final_output)



if __name__ == '__main__':
    main()
