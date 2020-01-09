def generate_heatmap(true_delta_matrix, label_noise_matrix, name):
    fig, ax = plt.subplots(1, 6, gridspec_kw={'width_ratios':[1, 0.08, 1, 0.08, 1, 0.08]}, figsize=(26, 8))
    plt.title(name)
    g1 =sns.heatmap(torch.abs(label_noise_matrix - true_delta_matrix), cmap="YlGnBu", cbar=True, ax = ax[0], cbar_ax= ax[1], vmin=0, vmax=0.5)
    g2 =sns.heatmap(true_delta_matrix, cmap="YlGnBu", cbar=True, ax=ax[2], cbar_ax=ax[3], vmin=0, vmax=1)
    g3 =sns.heatmap(label_noise_matrix, cmap="YlGnBu", cbar=True, ax= ax[4], cbar_ax=ax[5], vmin=0, vmax=1)

    ax[0].set_title("Estimation Error")
    ax[2].set_title("True Label Noise")
    ax[4].set_title("Estimated Label Noise")
    for axis in ax:
        tly = axis.get_yticklabels()
        axis.set_yticklabels(tly, rotation=0)
    plt.savefig('heatmaps/aug_{}.png'.format(name))

def plot_statistic_versus_aug_given_model(df, statistic, model_name):
    model_df = df.loc[df['Model Name'] == model_name]
    performance = model_df[statistic]
    y_pos = np.arange(len(model_df["Augmentations"]))
    plt.figure(figsize=(len(y_pos), 7))
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, model_df["Augmentations"], fontsize=15)
    plt.ylabel('{}'.format(statistic), fontsize=20)
    plt.xlabel('Augmentation', fontsize=20)
    plt.title(model_name, fontsize=30)
    xlocs, xlabs = plt.xticks()
    for i, v in enumerate(performance):
        plt.text(xlocs[i]-0.25, v + 0.01, str(round(v, 2)), fontsize=15)
    #plt.savefig('plots/{}_versus_aug/{}.pdf'.format(statistic, model_name))                                                                   
    plt.setp(xlabs, rotation=90, horizontalalignment='right')
    plt.show()
    plt.close('all')


def plot_statistic_std_dev_versus_aug_given_model(df, statistic, model_name):
    model_df = df.loc[df['Model Name'] == model_name]
    performance_mean = model_df["{} Mean".format(statistic)]
    performance_std = model_df["{} Std".format(statistic)]
    y_pos = np.arange(len(model_df["Augmentations"]))
    plt.figure(figsize=(len(y_pos), 7))
    plt.bar(y_pos, performance_mean, yerr=performance_std, align='center', alpha=0.5)
    plt.xticks(y_pos, model_df["Augmentations"], fontsize=15)
    plt.ylabel('{}'.format(statistic), fontsize =20)
    plt.xlabel('Augmentation', fontsize=20)
    plt.title(model_name, fontsize=30)
    xlocs, xlabs = plt.xticks()
    for i, v in enumerate(performance_mean):
        plt.text(xlocs[i] - 0.25, v + 0.01, str(round(v, 2)), fontsize=15)
    plt.setp(xlabs, rotation=30, horizontalalignment='right')
    plt.show()
    plt.close('all')
