import argparse
import yaml
import seaborn as sns
sns.reset_orig()
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc_file_defaults()
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
import distribution
from simulation import gen_data_dist, add_noise_to_class, gen_corrupted_labels
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
    n_classes = config["train"]["n_classes"] 
    if "clustering" in train["distribution"]:
        data_params = config["train"]
        for i in range(n_classes): 
            data_params["dist_{}".format(i)] = DISTRIBUTIONS["blob"]()

        assert(n_classes == train["distribution"]["clustering"]["centers"]) 
    else:
        data_params = config["train"]
        for i in range(n_classes): 
            dist_name = config["train"]["distribution"]["class_{}".format(i)]["name"]
    
            if config["train"]["distribution"]["class_{}".format(i)]["parameters"]:
                dist = DISTRIBUTIONS[dist_name](**config["train"]["distribution"]["class_{}".format(i)]["parameters"])
            else:
                dist = DISTRIBUTIONS[dist_name]()
            data_params["dist_{}".format(i)] = dist
        assert(n_classes == len(train["distribution"].keys())) 

    if config["plots"]:   
        delta_matrix = config["plots"]["delta_matrix"] 
        divergences = kl_divergence(data_params)
        generate_all_distribution_plots(data_params, delta_matrix, divergences)
    experiments = []

    if n_classes > 2:
        if config["experiment"]["multiclass_advanced"]:
           original_delta_matrix = []
           deltas = config["experiment"]["multiclass_advanced"]["deltas"]
           for i in range(len(deltas)):
               row_list = [deltas[i]/(n_classes-1) for i in range(n_classes)]
               row_list[i] = 1 - deltas[i]
               original_delta_matrix.append(row_list)
           original_delta_matrix = np.array(original_delta_matrix)
           multiclass_noise_advanced(data_params, original_delta_matrix)     
        if config["experiment"]["multiclass_simple"]:
           multiclass_noise_simple(data_params)     
    else:
        if config["experiment"]["class_noise_difference"]:
            print("Running class noise difference experiment")
            delta_stars = config["experiment"]["class_noise_difference"]["delta_stars"]
            auc_vs_class_noise_difference_experiment(data_params, delta_stars)
            experiments.append("class_noise_difference")
        if config["experiment"]["fixed_total_noise"]:
            print("Running fixed total noise experiment")
            total_noise = config["experiment"]["fixed_total_noise"]["noise"]  
            fixed_total_noise_experiment(data_params, total_noise)
            experiments.append("fixed_total_noise")
        if config["experiment"]["fix_delta0_vary_delta1"]:
            print("Running fixed delta0 vary delta1 experiment")
            delta0_range = config["experiment"]["fix_delta0_vary_delta1"]["delta0_range"]
            delta1_range = config["experiment"]["fix_delta0_vary_delta1"]["delta1_range"]
            fix_delta0_vary_delta1(data_params, delta0_range, delta1_range)
            experiments.append("fix_delta0_vary_delta1")
  
def get_data(data_params,delta_matrix):
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
    n_examples = data_params["n_examples"]
    n_runs = data_params["n_runs"]
    N_TRAIN = int(.5*n_examples)
    N_VAL = int(.25*n_examples)
    N_TEST = int(.25*n_examples)

    X, y = gen_data_dist(data_params, n_examples=n_examples)

    X_train, y_train = X[:N_TRAIN], y[:N_TRAIN]
    X_val, y_val = X[N_TRAIN:N_TRAIN+N_VAL], y[N_TRAIN:N_TRAIN+N_VAL]
    X_test, y_test = X[N_TRAIN+N_VAL:], y[N_TRAIN+N_VAL:]
    y_train_tildes = [gen_corrupted_labels(delta_matrix, y_train) for i in range(n_runs)]

    return [(X_train, y_train, y_train_tildes), (X_val,  y_val), (X_test, y_test)]

def score_delta(data_params, delta_matrix):
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

    train, val, test = get_data(data_params,delta_matrix)
    X_train, y_train, y_train_tildes = train
    X_val,  y_val = val
    X_test, y_test = test

    confusion_matrices = []    
    #Here we train a logistic regression model on each of the sets of corrupted labels
    scores = []
    for y_train_tilde in y_train_tildes:
        if len(delta_matrix) > 2:
            lr = LogisticRegression(solver='lbfgs', multi_class='multinomial', class_weight='balanced')
            lr.fit(X_train, y_train_tilde)
            y_pred = lr.predict(X_test)
            scores.append(f1_score(y_test, y_pred, average='weighted') *1.0)
            confusion_matrices.append(confusion_matrix(y_test, y_pred))
        else:
            lr = LogisticRegression(solver='lbfgs', class_weight='balanced')
            lr.fit(X_train, y_train_tilde)
            y_pred = lr.predict(X_test)
            scores.append(roc_auc_score(y_test, y_pred))
            confusion_matrices.append(confusion_matrix(y_test, y_pred))

    avg_confusion = np.mean(np.array(confusion_matrices), axis=0)
    return sum(scores)/len(scores), np.std(np.array(scores)), avg_confusion 

def scores_deltas(data_params, delta_vals):
    """
    A wrapper method for the score_delta function.
    Calls the score_delta function for all pairs of
    (delta_0, delta_1) in delta_vals.
    
    Parameters:
        delta_vals: a list of (delta_0, delta_1) pairs.
        n_runs: number of trials to run per (delta_0, delta_1) pair.
    """
    scores = []
    std_devs = []
    confusion_matrices = []
    for delta_0, delta_1 in delta_vals:
        delta_matrix = [[1-delta_0, delta_0], [delta_1, 1-delta_1]]
        res = score_delta(data_params,delta_matrix)

        scores.append(res[0])
        std_devs.append(res[1])
        confusion_matrices.append(res[2])
    return scores, std_devs, confusion_matrices

def kl_divergence(data_params):
    divergences = {}
    n_classes = data_params["n_classes"]
    grid = np.mgrid[-10:10:0.5, -10:10:0.5].reshape(2,-1).T
    for i in range(n_classes):
        for j in range(n_classes):
            if i !=j:
                dist0 = data_params["dist_{}".format(i)]
                dist1 = data_params["dist_{}".format(j)]
                p = dist0.pdf(grid)
                q = dist1.pdf(grid)
                divergence = round(np.sum(np.where(p != 0, p * np.log(p / q), 0)), 2)
                divergences["D(dist{}|dist{})".format(i, j)] = divergence

    s = "KL Divergence:"
    for i in divergences:
        s+= "{}: {}\n".format(i, divergences[i])
    return s

def auc_vs_class_noise_difference_experiment(data_params, delta_stars):
    """
    We observe that AUC deteriorates as the difference between class noise levels increases.
    We fix delta_star as the mean probability of flipping a class 1 or class 0 label.
    We define the probability of flipping class-0 labels as delta_star - delta. 
    We define the probability of flipping class-1 labels as delta_star + delta.
    """
    fig = plt.figure(0)
    for i, delta_star in enumerate(delta_stars):
        deltas = [(delta_star-delta/2, delta_star+delta/2) for delta in np.linspace(0, delta_star, 20)]
        scores = scores_deltas(data_params, deltas)
        plt.errorbar(np.linspace(0, delta_star, 20), scores[0], yerr=scores[1], label="delta_star={}".format(delta_star))
    plt.xlim(0, .4)
    plt.ylim(0, 1)
    plt.xlabel("Noise Rate Difference")
    plt.ylabel("AUC")
    plt.title("Various Delta* | AUC vs. Class Noise Difference")
    plt.legend()
    plt.savefig("plots/class_noise_difference/dist0_{}_dist1_{}.pdf".format(data_params["dist_0"].info(), data_params["dist_1"].info()))

def multiclass_noise_advanced(data_params, original_delta_matrix):
    """
    """
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
    noise_amounts = np.linspace(0, original_delta_matrix[row_with_max_noise][row_with_max_noise], 30)

    scores = []
    std_devs = []
    for i in range(len(noise_amounts)):
        new_delta_matrix = original_delta_matrix + noise_amounts[i] * noise_addition_matrix
        score, std, confusion_matrices = score_delta(data_params, new_delta_matrix)
        scores.append(score)
        std_devs.append(std)

    ymax = max(scores)
    xpos = scores.index(ymax)
    xmax = noise_amounts[xpos]

    plt.errorbar(noise_amounts, scores, yerr=std_devs, label="nonuniform")
    plt.annotate('max: {}'.format(ymax), xy=(xmax, ymax), xytext=(xmax, ymax+5),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )        
    plt.xlim(0, noise_amounts[-1])
    plt.ylim(0, 1)
    plt.xlabel("Noise added")
    deltas = [round(1-original_delta_matrix[i][i],1) for i in range(len(original_delta_matrix))]
    plt.title("Multiclass Label Perturbation- Deltas={}".format(deltas))
    plt.legend()
    plt.savefig("multiclass/deltas_{}.png".format(deltas))


def multiclass_noise_simple(data_params):
    """
    """
    noise_amounts = np.linspace(0, 1, 30)

    scores = []
    std_devs = []
    for i in range(len(noise_amounts)):
        original_delta_matrix = np.identity(data_params["n_classes"])
        original_delta_matrix[0][0] = 1- noise_amounts[i]
        original_delta_matrix[0][1] = noise_amounts[i]
        for j in range(data_params["n_classes"]):
            original_delta_matrix[1][j] = 0.15
        original_delta_matrix[1][1] = 0.55
        score, std, confusion_matrices = score_delta(data_params, original_delta_matrix)
        scores.append(score)
        std_devs.append(std)

    xmax = noise_amounts[xpos]

    plt.errorbar(noise_amounts, scores, yerr=std_devs, label="nonuniform")
    plt.annotate('max: {}'.format(ymax), xy=(xmax, ymax), xytext=(xmax, ymax+5),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )        
    plt.xlim(0, noise_amounts[-1])
    plt.ylim(0, 1)
    plt.xlabel("Noise added")
    plt.title("Multiclass Label Perturbation- Vary Delta0")
    plt.legend()
    plt.savefig("multiclass/simple.png")

def fixed_total_noise_experiment(data_params, total_noise_vals):
    """
    We fix the "total amount of noise" which is the sum of the label flipping rate for both classes. 
    For instance, set delta_0 + delta_1 = 0.5. We examine a variety of different (delta_0, delta_1) 
    combinations to see which of them results in the highest test accuracy.
    """
    fig = plt.figure(1)

    for total_noise in total_noise_vals:
        delta_vals = [(delta_0, total_noise-delta_0) for delta_0 in np.linspace(0, total_noise, 20)]
        delta_0_coords = [delta_0 for (delta_0, delta_1) in delta_vals]
        scores = scores_deltas(data_params, delta_vals)
        plt.errorbar(delta_0_coords, scores[0], yerr=scores[1], label='total_noise={}'.format(total_noise))
    plt.ylim(0, 1)
    plt.xlim(0, 0.8)
    plt.xlabel("Delta_0")
    plt.ylabel("AUC")
    plt.title("Fixed Total Noise| AUC vs. Delta_0".format(total_noise))
    plt.legend()
    plt.savefig("plots/fixed_total_noise/dist0_{}_dist1_{}.pdf".format(data_params["dist_0"].info(), data_params["dist_1"].info()))

def fix_delta0_vary_delta1(data_params, delta0_range, delta1_range):
    """
    In this experiment, we fix delta_0 and find the optimal amount of noise to inject in delta_1.
    """
    delta_0_coords = np.linspace(delta0_range[0], delta0_range[1], 5)
    delta_1_coords = np.linspace(delta1_range[0], delta1_range[1], 20)
    delta_vals = [[(delta_0, delta_1) for delta_1 in delta_1_coords] for delta_0 in delta_0_coords]

    scores = [scores_deltas(data_params, subset) for subset in delta_vals]
    fig = plt.figure(2)
    labels = []
    for i in range(len(delta_0_coords)):
        labels.append('delta_0={}'.format(round(delta_0_coords[i], 2)))
        plt.errorbar(delta_1_coords, scores[i][0], yerr=scores[i][1], label= labels[-1])
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.xlabel("Delta_1")
    plt.ylabel("AUC")
    plt.title("Fixed Delta_0, Varied Delta_1 | AUC vs. Delta_1")
    plt.legend()
    plt.savefig("plots/fix_delta0_vary_delta1/dist0_{}_dist1_{}.pdf".format(data_params["dist_0"].info(), data_params["dist_1"].info()))
    plt.close()

def generate_all_distribution_plots(data_params, delta_matrix, divergences):
    train, val, test = get_data(data_params, delta_matrix)

    X_train, y_train, y_train_tildes = train
    X_test, y_test = test
    y_train_tilde = y_train_tildes[0]
    lr = LogisticRegression(solver='lbfgs', multi_class="multinomial")
    lr.fit(X_train, y_train_tilde)
    y_pred = lr.predict(X_test)
    lr2 = LogisticRegression(solver='lbfgs', multi_class="multinomial")
    lr2.fit(X_train, y_train)
    y_pred_noiseless = lr2.predict(X_test)
 
    f, axes = plt.subplots(2, 3, figsize=(14, 7))

    sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=y_train, 
                    ax=axes[0, 0])
    axes[0, 0].set_title("Noiseless Train Distribution")
    sns.scatterplot(x=X_train[:,0], y=X_train[:, 1], hue=y_train_tilde, 
                    ax=axes[0, 1])
    axes[0, 1].set_title("Noisy Train Distribution")
    sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=y_test, 
                    ax=axes[1, 2])
    axes[1, 2].set_title("True Test Distribution")
    sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=y_pred, 
                    ax=axes[1,1])
    axes[1, 1].set_title("Trained on Noisy, Predicted Test Distribution")
    sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=y_pred_noiseless, 
                    ax=axes[1,0])
    axes[1, 0].set_title("Trained on Clean, Predicted Test Distribution")
    
    axes[0, 2].text(0.5, 0.01, divergences, wrap=True, horizontalalignment='center', fontsize=12) 
    f.savefig("plots/distributions/{}.png".format(summarize_data_params(data_params)))    
    plt.close()
def summarize_data_params(data_params):
    if "clustering" in data_params:
        return "blobs_{}_centers".format(data_params["clustering"]["centers"])
    else:
        s = ""
        for i in range(data_params["n_classes"]):
             s += "dist{}_{}".format(i, data_params["dist_{}".format(i)].info())

        return s
if __name__ == "__main__":
     main()
