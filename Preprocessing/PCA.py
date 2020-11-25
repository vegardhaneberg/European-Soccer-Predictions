from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

from Preprocessing import GetData


def pca(data, target_feature, components=2, plot=False):
    """
    function that performs a pca dimensionality reduction on the input dataframe
    :param data: dataframe with preprocessed data
    :param target_feature: the target that the models are predicting.
    :param components: number of pca components
    :param plot: bolean. If true the transformation is plotted, if false it is not.
    :return: dataframe that is pca transformed
    """
    print('Starting principal components analysis...')

    features = []

    column_names = []
    for i in range(1, components + 1):
        column_names.append('pc ' + str(i))

    for feature in data:
        if feature != target_feature:
            features.append(feature)

    x = data.loc[:, features].values
    y = data.loc[:, [target_feature]].values

    x = StandardScaler().fit_transform(x)

    pca_performed = PCA(n_components=components)

    principal_components = pca_performed.fit_transform(x)
    principal_df = pd.DataFrame(data=principal_components, columns=column_names)
    final_df = pd.concat([principal_df, data[[target_feature]]], axis=1)

    if components == 2 and plot:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_title('PCA With 2 Components', fontsize=20)
        targets = [0, 1, 2]
        colors = ['navajowhite', 'palevioletred', 'indigo']
        for target, color in zip(targets, colors):
            indices_to_keep = final_df[target_feature] == target
            ax.scatter(final_df.loc[indices_to_keep, 'pc 1']
                       , final_df.loc[indices_to_keep, 'pc 2']
                       , c=color
                       , s=40)
        ax.legend(
            ['Home win', 'Draw', 'Away win'], fontsize=17)
        plt.show()

    return final_df
