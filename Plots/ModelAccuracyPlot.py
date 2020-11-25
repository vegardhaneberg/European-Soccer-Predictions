import matplotlib.pyplot as plt
import seaborn as sns


def present_results(model_names, accuracies):
    """
    function that prints the accuracies of the models as a bar chart.
    :param model_names: a list with the model names as strings
    :param accuracies: a list with the accuracies as floats
    """
    plt.rcParams['figure.figsize'] = 15, 6
    sns.set_style("darkgrid")
    ax = sns.barplot(x=model_names, y=accuracies, palette="rocket", saturation=1.5)
    #plt.xlabel("Classifier Models", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.title("Accuracy of the Classifier Models", fontsize=20)
    plt.xticks(fontsize=12, horizontalalignment='center', rotation=0)
    plt.yticks(fontsize=13)
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        ax.annotate(f'{height:.2%}', (x + width/2, y + height*1.02), ha='center', fontsize='x-large')
    plt.show()
