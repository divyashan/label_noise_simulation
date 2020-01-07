from evaluate import generate_heatmap, plot_accuracy_with_diff_augmentations, plot_kl_with_diff_augmentations
import torch



def test_generate_heatmap():
    true_delta_matrix = torch.eye(10)
    label_noise_matrix = torch.eye(10) * 0.9
    name = "test"

    generate_heatmap(true_delta_matrix, label_noise_matrix, name)

def test_generate_aug_acc_plot():
    list_of_augmentations= [[], ['rot_1'], ['rot_2'], ['rot_4'], ['rot_8'], ['rot_16']]
    accuracy_given_auglist = [[0.5683, 0.5677, 0.6214, 0.5877], [0.5788, 0.5769, 0.6254, 0.6002], [0.5823, 0.5802, 0.6309, 0.6031], [0.5875, 0.5884, 0.6333, 0.6114], [0.5908, 0.5903, 0.6363, 0.6148], [0.5946, 0.5946, 0.6375, 0.6181]]
    models_names = ['cifar_clean', 'cifar_swap2', 'cifar_swap3', 'cifar_swap5']
    plot_accuracy_with_diff_augmentations(list_of_augmentations, accuracy_given_auglist, models_names)

def test_generate_aug_kl_plot():
    list_of_augmentations= [[], ['rot_1'], ['rot_2'], ['rot_4'], ['rot_8'], ['rot_16']]
    kl_dic = {"mean": [5, 4, 3, 2], "std": [1, 1, 1, 1]}
    kl_dic2 = {"mean": [4, 3, 2, 1], "std": [2, 2, 2, 2]}
    kl_all = [kl_dic, kl_dic2, kl_dic, kl_dic2, kl_dic, kl_dic2]
    models_names = ['cifar_clean', 'cifar_swap2', 'cifar_swap3', 'cifar_swap5']
    plot_kl_with_diff_augmentations(list_of_augmentations, kl_all, models_names)


test_generate_aug_kl_plot()

