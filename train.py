import argparse
import yaml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns
import distribution
from simulation import gen_data_dist, add_noise_to_class, gen_corrupted_labels
sns.set_style('whitegrid')
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

    n_examples = config["train"]["n_examples"]
    class_split = config["train"]["class_split"]
    n_runs = config["train"]["n_runs"] 
    
    if "clustering" in config["train"]["distribution"]:
        data_params = config["train"]["distribution"] 
        data_params["dist_0"] = DISTRIBUTIONS["blob"]() 
        data_params["dist_1"] = DISTRIBUTIONS["blob"]()
    else: 
        dist_0_name = config["train"]["distribution"]["class_0"]["name"]
        dist_1_name = config["train"]["distribution"]["class_1"]["name"]
    
        if config["train"]["distribution"]["class_0"]["parameters"]:
            dist_0 = DISTRIBUTIONS[dist_0_name](**config["train"]["distribution"]["class_0"]["parameters"])
        else:
            dist_0 = DISTRIBUTIONS[dist_0_name]()

        if config["train"]["distribution"]["class_1"]["parameters"]:
            dist_1 = DISTRIBUTIONS[dist_1_name](**config["train"]["distribution"]["class_1"]["parameters"])
        else:
            dist_1 = DISTRIBUTIONS[dist_1_name]()
        data_params = {"dist_0": dist_0, "dist_1": dist_1, "class_split": class_split}
    generate_all_distribution_plots(data_params, n_examples)   

    experiments = []
    if config["experiment"]["class_noise_difference"]:
        print("Running class noise difference experiment")
        delta_stars = config["experiment"]["class_noise_difference"]["delta_stars"]
        auc_vs_class_noise_difference_experiment(data_params,  n_examples, n_runs, delta_stars)
        experiments.append("class_noise_difference")
    if config["experiment"]["fixed_total_noise"]:
        print("Running fixed total noise experiment")
        total_noise = config["experiment"]["fixed_total_noise"]["noise"]  
        fixed_total_noise_experiment(data_params, n_examples, n_runs, total_noise)
        experiments.append("fixed_total_noise")
    if config["experiment"]["fix_delta0_vary_delta1"]:
        print("Running fixed delta0 vary delta1 experiment")
        delta0_range = config["experiment"]["fix_delta0_vary_delta1"]["delta0_range"]
        delta1_range = config["experiment"]["fix_delta0_vary_delta1"]["delta1_range"]
        fix_delta0_vary_delta1(data_params, n_examples, n_runs, delta0_range, delta1_range)
        experiments.append("fix_delta0_vary_delta1")
   
def get_data(data_params, n_examples, n_runs, delta_0, delta_1):
    """
    Method that returns three datasets
        1. Training set with corrupted and n_runs sets of 
           corrupted labels
        2. Validation set (uncorrupted)
        3. Test set (uncorrupted)
        
    Parameters:
        n_examples: length of total dataset
        n_runs: number of trials to run per (delta_0, delta_1) pair.
        delta_0: the probability that we flip labels that are equal to 0
        delta_1: the probability that we flip labels that are equal to 1
    """
    N_TRAIN = int(.5*n_examples)
    N_VAL = int(.25*n_examples)
    N_TEST = int(.25*n_examples)

    X, y = gen_data_dist(data_params, n_examples=n_examples)

    X_train, y_train = X[:N_TRAIN], y[:N_TRAIN]
    X_val, y_val = X[N_TRAIN:N_TRAIN+N_VAL], y[N_TRAIN:N_TRAIN+N_VAL]
    X_test, y_test = X[N_TRAIN+N_VAL:], y[N_TRAIN+N_VAL:]
    y_train_tildes = [gen_corrupted_labels(delta_0, delta_1, y_train) for i in range(n_runs)]

    return [(X_train, y_train, y_train_tildes), (X_val,  y_val), (X_test, y_test)]

def score_delta(data_params, n_examples, n_runs, delta_0, delta_1):
    """
    Method that returns the average score of a Logistic
    Regression model on a test set where the training data has
    been corrupted by flipping class-1 labels of the training 
    set with probability delta_1 and flipping class-0 labels of
    the test set with probability delta_0.
    
    Parameters:
        n_examples: length of the dataset
        n_runs: number of trials to run per (delta_0, delta_1) pair.
        delta_0: the probability that we flip labels that are equal to 0
        delta_1: the probability that we flip labels that are equal to 1
    """
    train, val, test = get_data(data_params, n_examples, n_runs, delta_0, delta_1)
    X_train, y_train, y_train_tildes = train
    X_val,  y_val = val
    X_test, y_test = test
    
    # Noisy LR accuracy
    avg_score = 0
    
    #Here we train a logistic regression model on each of the sets of corrupted labels
    for y_train_tilde in y_train_tildes:
        lr = LogisticRegression(solver='lbfgs')
        lr.fit(X_train, y_train_tilde)
        avg_score += lr.score(X_test, y_test)*1.0
    avg_score/=n_runs
    return avg_score

def scores_deltas(data_params, delta_vals, n_examples, n_runs):
    """
    A wrapper method for the score_delta function.
    Calls the score_delta function for all pairs of
    (delta_0, delta_1) in delta_vals.
    
    Parameters:
        delta_vals: a list of (delta_0, delta_1) pairs.
        n_runs: number of trials to run per (delta_0, delta_1) pair.
    """
    scores = []
    for delta_0, delta_1 in delta_vals:
            scores.append(score_delta(data_params, n_examples, n_runs, delta_0, delta_1))
    return scores

def auc_vs_class_noise_difference_experiment(data_params, n_examples, n_runs, delta_stars):
    """
    We observe that AUC deteriorates as the difference between class noise levels increases.
    We fix delta_star as the mean probability of flipping a class 1 or class 0 label.
    We define the probability of flipping class-0 labels as delta_star - delta. 
    We define the probability of flipping class-1 labels as delta_star + delta.
    """
    fig = plt.figure(0)
    for delta_star in delta_stars:
        deltas = [(delta_star-delta/2, delta_star+delta/2) for delta in np.linspace(0, delta_star, 20)]
        scores = scores_deltas(data_params, deltas, n_examples, n_runs)
        plt.plot(np.linspace(0, delta_star, 20), scores, label="{}".format(delta_star))
    plt.xlim(0, .4)
    plt.ylim(0, 1)
    plt.xlabel("Noise Rate Difference")
    plt.ylabel("AUC")
    plt.title("Various Delta* | AUC vs. Class Noise Difference")
    plt.legend()
    plt.savefig("plots/class_noise_difference/dist0_{}_dist1_{}.pdf".format(data_params["dist_0"].info(), data_params["dist_1"].info()))

def fixed_total_noise_experiment(data_params, n_examples, n_runs,  total_noise_vals):
    """
    We fix the "total amount of noise" which is the sum of the label flipping rate for both classes. 
    For instance, set delta_0 + delta_1 = 0.5. We examine a variety of different (delta_0, delta_1) 
    combinations to see which of them results in the highest test accuracy.
    """
    fig = plt.figure(1)

    for total_noise in total_noise_vals:
        delta_vals = [(delta_0, total_noise-delta_0) for delta_0 in np.linspace(0, total_noise, 20)]
        delta_0_coords = [delta_0 for (delta_0, delta_1) in delta_vals]
        scores = scores_deltas(data_params, delta_vals, n_examples, n_runs)
        fixed_total_noise_scores = scores
        plt.plot(delta_0_coords, fixed_total_noise_scores, label='total_noise={}'.format(total_noise))
    plt.ylim(0, 1)
    plt.xlim(0, 0.8)
    plt.xlabel("Delta_0")
    plt.ylabel("AUC")
    plt.title("Fixed Total Noise| AUC vs. Delta_0".format(total_noise))
    plt.legend()
    plt.savefig("plots/fixed_total_noise/dist0_{}_dist1_{}.pdf".format(data_params["dist_0"].info(), data_params["dist_1"].info()))

def fix_delta0_vary_delta1(data_params, n_examples, n_runs, delta0_range, delta1_range):
    """
    In this experiment, we fix delta_0 and find the optimal amount of noise to inject in delta_1.
    """
    delta_0_coords = np.linspace(delta0_range[0], delta0_range[1], 5)
    delta_1_coords = np.linspace(delta1_range[0], delta1_range[1], 20)
    delta_vals = [[(delta_0, delta_1) for delta_1 in delta_1_coords] for delta_0 in delta_0_coords]

    scores = [scores_deltas(data_params, subset, n_examples, n_runs) for subset in delta_vals]
    fig = plt.figure(2)
    for i in range(len(delta_0_coords)):
        plt.plot(delta_1_coords, scores[i], label= 'delta_0={}'.format(delta_0_coords[i]))
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.xlabel("Delta_1")
    plt.ylabel("AUC")
    plt.title("Fixed Delta_0, Varied Delta_1 | AUC vs. Delta_1")
    plt.legend()
    plt.savefig("plots/fix_delta0_vary_delta1/dist0_{}_dist1_{}.pdf".format(data_params["dist_0"].info(), data_params["dist_1"].info()))

def generate_all_distribution_plots(data_params, n_examples, delta_0=0.4, delta_1=0):
    train, val, test = get_data(data_params, n_examples, 1, delta_0, delta_1)

    X_train, y_train, y_train_tildes = train
    X_test, y_test = test
    y_train_tilde = y_train_tildes[0]
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(X_train, y_train_tilde)
    y_pred = lr.predict(X_test)
    score = lr.score(X_test, y_test)
    f, axes = plt.subplots(2, 2, figsize=(7, 7))

    sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=y_train, palette="Set2", 
                    ax=axes[0, 0])
    axes[0, 0].set_title("Noiseless Train Distribution")
    sns.scatterplot(x=X_train[:,0], y=X_train[:, 1], hue=y_train_tilde, palette="Set2",
                    ax=axes[0, 1])
    axes[0, 1].set_title("Noisy Train Distribution")
    sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=y_test, palette="Set2",
                    ax=axes[1, 0])
    axes[1, 0].set_title("True Test Distribution")
    sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=y_pred, palette="Set2",
                    ax=axes[1,1])
    axes[1, 1].set_title("Predicted Test Distribution")
    
    f.savefig("plots/distributions/dist0_{}_dist1_{}.png".format(data_params["dist_0"].info(), data_params["dist_1"].info()))    
   
if __name__ == "__main__":
     main()
#     generate_train_noiseless_plot(data_params)
