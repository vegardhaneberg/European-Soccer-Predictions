from matplotlib import pyplot as plt
from sklearn import metrics


def plot_accuracy(history, model_name):
    """
    function that plots the accuracy of the training and validation data for a neural network
    :param history: the return from the model.fot method
    :param model_name: string indicating the name of the model
    """
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(model_name + ' Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training Data', 'Validation Data'], loc='upper left')
    plt.show()


def plot_loss(history, model_name):
    """
    function that plots the loss of the training and validation data for a neural network
    :param history: the return from the model.fot method
    :param model_name: string indicating the name of the model
    """
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(model_name + ' Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training Data', 'Validation Data'], loc='upper right')
    plt.show()


def get_confusion_matrix(y_pred, y_test):
    """
    function that calculates the confusion matrix
    :param y_pred: predicted values
    :param y_test: true values
    :return: confusion matrix
    """
    y_pred_converted = []
    for list in y_pred:
        index = list.index(max(list))
        y_pred_converted.append(index)

    y_test_converted = []
    for list in y_test:
        index = list.index(max(list))
        y_test_converted.append(index)

    return metrics.confusion_matrix(y_test_converted, y_pred_converted)


def get_percentage_differences(y_pred, y_test):
    """
    function that calculates the differences in the predictions between home win and away win for a neural network
    :param y_pred: predicted values, as probabilities
    :param y_test: true values
    :return: a list where the first element is the average difference between home win and away win predicted by
    the neural network, when the match was a home win. The second is the same, but when the match outcome was a draw and
    the third element for away win
    """
    diffs_home = []
    diffs_draw = []
    diffs_away = []

    for pos in range(len(y_test)):
        if y_test[pos][0] == 1:
            diffs_home.append(abs(y_pred[pos][0] - y_pred[pos][2]))
        if y_test[pos][1] == 1:
            diffs_draw.append(abs(y_pred[pos][0] - y_pred[pos][2]))
        if y_test[pos][2] == 1:
            diffs_away.append(abs(y_pred[pos][0] - y_pred[pos][2]))

    return sum(diffs_home) / (len(diffs_home)), sum(diffs_draw) / (len(diffs_draw)), sum(diffs_away) / (len(diffs_away))
