from sklearn.metrics import accuracy_score
from sklearn import svm
from matplotlib import pyplot as plt
import time
import numpy as np
from matplotlib.colors import Normalize
from sklearn import metrics


def support_vector_machine(x_train, x_test, y_train, y_test, C, gamma):
    """
    function that trains a support vector machine and returns the accuracy. The confusion matrix is printed
    :param x_train: input training data
    :param x_test: input test data
    :param y_train: output test data
    :param y_test: output test data
    :param C: regularization factor
    :param gamma: width of the kernel
    :return: the accuracy of the model
    """
    clf = svm.SVC(kernel='rbf', C=C, gamma=gamma)
    clf.fit(x_train, y_train)

    y_predict = clf.predict(x_test)
    acc = accuracy_score(y_test, y_predict)

    print(metrics.confusion_matrix(y_test, y_predict))
    return acc


def grid_search(x_train, x_test, y_train, y_test):
    """
    function that performs a grid search for the svm to find the optimal c and gamma. The best c and gamma is printed.
    The functino alsp plots the different pairs of c and gamma as a heatmap
    :param x_train: input training data
    :param x_test: input test data
    :param y_train: output test data
    :param y_test: output test data
    """
    accuracy_grid = []
    max_acc = 0
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 0, 10)
    for i, C in enumerate(C_range):
        slow = False
        accuracy_grid.append([])
        for gamma in gamma_range:
            if slow:
                accuracy_grid[i].append(0.45)
            else:
                tic = time.perf_counter()
                acc = support_vector_machine(x_train, x_test, y_train, y_test, C, gamma)
                toc = time.perf_counter()
                accuracy_grid[i].append(acc)
                if acc > max_acc:
                    max_acc = acc
                    max_C = C
                    max_gamma = gamma
                if toc - tic > 40:
                    slow = True
                print('C =', C, 'gamma =', gamma, 'Acc = ', acc, 'Took', toc - tic, 'seconds (', i, '/ 13)')

    print('Max is', max_acc, 'with C =', max_C, 'and gamma =', max_gamma)
    visualize_svm_hyperparameters(accuracy_grid, 'SVM_grid.png')


def visualize_svm_hyperparameters(accuracy_matrix, filename):
    """
    function that creates a heat map. used to visualize the results of the grid search
    :param accuracy_matrix: the accuracies for the svm
    :param filename: filename to save the heatmap
    """
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(accuracy_matrix, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.45, midpoint=0.6))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.title('Validation accuracy')
    plt.savefig(filename)


class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
