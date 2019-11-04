import argparse
import yaml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
import distribution
from simulation import gen_data_dist, add_noise_to_class, gen_corrupted_labels
from train import get_data, multiclass_noise_advanced
import cleanlab
import os

DEFAULT_CONFIG = os.path.dirname(__file__) + "configs/simulated_noise.yaml" 
DISTRIBUTIONS = {"gaussian": distribution.MultivariateGaussian,
                 "exponential": distribution.Exponential,
                 "geometric": distribution.Geometric,
                 "uniform": distribution.Uniform,
                 "blob": distribution.Blob}


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

    train = config["train"]
    n_classes = train["n_classes"]
    n_examples = train["n_examples"]
    class_split = train["class_split"]
    n_runs = train["n_runs"] 

     
    if "clustering" in train["distribution"]:
        data_params = config["train"]["distribution"]
        data_params["class_split"] =  class_split
        data_params["n_classes"] =  n_classes
        for i in range(n_classes): 
            data_params["dist_{}".format(i)] = DISTRIBUTIONS["blob"]()

        assert(n_classes == train["distribution"]["clustering"]["centers"]) 
    else:
        data_params = {}
        data_params["class_split"] =  class_split
        data_params["n_classes"] =  n_classes
 
        for i in range(n_classes): 
            dist_name = config["train"]["distribution"]["class_{}".format(i)]["name"]
    
            if config["train"]["distribution"]["class_{}".format(i)]["parameters"]:
                dist = DISTRIBUTIONS[dist_name](**config["train"]["distribution"]["class_{}".format(i)]["parameters"])
            else:
                dist = DISTRIBUTIONS[dist_name]()
            data_params["dist_{}".format(i)] = dist
        assert(n_classes == len(train["distribution"].keys())) 
   
    delta_matrix = config["plots"]["delta_matrix"] 

    name, res= learn_with_noisy_labels(data_params, n_examples, n_runs, delta_matrix)
    comparison_plots(name, res)
    avg_noise_matrix = rp_to_estimate_noise(data_params, n_examples, n_runs, delta_matrix)
    print(avg_noise_matrix)
    multiclass_noise_advanced(avg_noise_matrix, data_params, n_examples, n_runs) 

 
def learn_with_noisy_labels(data_params, n_examples, n_runs, delta_matrix):
    train, val, test = get_data(data_params, n_examples, n_runs, delta_matrix)  
    X_train, y_train, y_train_tildes = train
    X_val,  y_val = val
    X_test, y_test = test

    rp_scores = []
    baseline_noisy_scores = []
    baseline_clean_scores = []
    for y_train_tilde in y_train_tildes:
        lnl = cleanlab.classification.LearningWithNoisyLabels(clf=LogisticRegression(solver='lbfgs', multi_class='multinomial', class_weight='balanced'))
        lnl.fit(X_train, y_train_tilde)
        y_pred = lnl.predict(X_test)
        rp_scores.append(f1_score(y_test, y_pred, average='weighted')*1.0)

        lr = LogisticRegression(solver='lbfgs', multi_class='multinomial', class_weight='balanced')
        lr.fit(X_train, y_train_tilde)
        y_pred = lr.predict(X_test)
        baseline_noisy_scores.append(f1_score(y_test, y_pred, average='weighted')*1.0)
 
    lr = LogisticRegression(solver='lbfgs', multi_class='multinomial', class_weight='balanced')
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    baseline_clean_scores.append(f1_score(y_test, y_pred, average='weighted')*1.0)

    scores = [rp_scores, baseline_clean_scores, baseline_noisy_scores]
    name = ["rp", "baseline_clean", "baseline_noisy"]
    res = []
    for sc in scores:
        res.append((sum(sc)/len(sc), np.std(np.array(sc))))
    return name, res


def rp_to_estimate_noise(data_params, n_examples, n_runs, delta_matrix):
    train, val, test = get_data(data_params, n_examples, n_runs, delta_matrix)  
    X_train, y_train, y_train_tildes = train
    X_val,  y_val = val
    X_test, y_test = test
    noise_matrices = []

    for y_train_tilde in y_train_tildes:
        lnl = cleanlab.classification.LearningWithNoisyLabels(clf=LogisticRegression(solver='lbfgs', multi_class='multinomial', class_weight='balanced'))
        lnl.fit(X_train, y_train_tilde)
        y_pred = lnl.predict(X_test)
        y_train_pred = lnl.predict(X_train) 

        lr = LogisticRegression(solver='lbfgs', multi_class='multinomial', class_weight='balanced')
        lr.fit(X_train, y_train_tilde)
        y_pred = lr.predict(X_test)
        y_pred_proba = lr.predict_proba(X_train)
        noise_matrix = estimate_noise_matrix(X_train, y_train_tilde, y_train_pred, y_pred_proba)  
        noise_matrices.append(noise_matrix)
 
    avg_noise_matrix = np.mean(noise_matrices, axis=0) 
    return avg_noise_matrix


def estimate_noise_matrix(X_train, y_train_tilde, y_train_pred, y_pred_proba):
    label_errors = cleanlab.pruning.get_noise_indices(
        s=y_train_tilde, # required
        psx=y_pred_proba # required
    )
    noise_matrix = [[0 for i in y_pred_proba[0]] for j in y_pred_proba[0]]
 
    for i, error in enumerate(label_errors):
        noise_matrix[int(y_train_pred[i])][int(y_train_tilde[i])] += 1
    for i in range(len(noise_matrix)):
        subarr_sum = sum(noise_matrix[i])
        for j in range(len(noise_matrix)):
            noise_matrix[i][j] /= subarr_sum
    return noise_matrix
    
def construct_noise_addition(original_delta_matrix):
    flip_correct_probs = [original_delta_matrix[i][i] for i in range(len(original_delta_matrix))]
    row_with_max_noise = flip_correct_probs.index(min(flip_correct_probs))

    noise_addition_matrix = []
    for row in range(len(original_delta_matrix)):
        if row == row_with_max_noise:
            row_list = [0 for i in range(len(original_delta_matrix))]
        else:
            row_list = [1-original_delta_matrix[row][i] for i in range(len(original_delta_matrix))]
            row_list[row] = -3 + sum(original_delta_matrix[row]) - original_delta_matrix[row][row]
        noise_addition_matrix.append(row_list)
    noise_addition_matrix = np.array(noise_addition_matrix)
    return noise_addition_matrix
 
def comparison_plots(name, res):
    for i in range(len(name)):
        print(name[i], res[i])

if __name__ == "__main__":
     main()
