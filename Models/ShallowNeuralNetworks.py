from tensorflow import keras
from Plots import NeuralNetworkHistoryPlot


def shallow_neural_network_relu(x_train, x_test, y_train, y_test, number_of_features):
    """
    function that trains a neural network with one hidden layer and relu activation. The function prints the
    confusion matrix for the model and the training plot over its epochs
    :param x_train: input training data
    :param x_test: input test data
    :param y_train: output test data
    :param y_test: output test data
    :param number_of_features: number of input features to the network. Equal to the number of neurins in the first
    layer
    :return: the accuracy of the model
    """

    model = keras.Sequential([
        keras.Input(shape=(number_of_features,)),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='sgd', loss=keras.losses.CategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

    validation_splitpoint = int(len(x_test) * 0.1)
    x_val = x_train[:validation_splitpoint]
    x_train = x_train[validation_splitpoint:]
    y_val = y_train[:validation_splitpoint]
    y_train = y_train[validation_splitpoint:]

    history = model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val))

    loss, acc = model.evaluate(x_test, y_test)

    NeuralNetworkHistoryPlot.plot_accuracy(history, 'Shallow NN Relu')
    NeuralNetworkHistoryPlot.plot_loss(history, 'Shallow NN Relu')

    y_pred = model.predict(x_test).tolist()
    confusion_matrix = NeuralNetworkHistoryPlot.get_confusion_matrix(y_pred, y_test)
    home_diff, draw_diff, away_diff = NeuralNetworkHistoryPlot.get_percentage_differences(y_pred, y_test)
    print('Confusion Matrix:')
    print(confusion_matrix)
    print('Diffs')
    print(home_diff, draw_diff, away_diff)

    return acc


def shallow_neural_network_tanh(x_train, x_test, y_train, y_test, number_of_features):
    """
    function that trains a neural network with one hidden layer and tanh activation. The function prints the
    confusion matrix for the model and the training plot over its epochs
    :param x_train: input training data
    :param x_test: input test data
    :param y_train: output test data
    :param y_test: output test data
    :param number_of_features: number of input features to the network. Equal to the number of neurins in the first
    layer
    :return: the accuracy of the model
    """

    model = keras.Sequential([
        keras.Input(shape=(number_of_features,)),
        keras.layers.Dense(10, activation='tanh'),
        keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='sgd', loss=keras.losses.CategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

    validation_splitpoint = int(len(x_test) * 0.1)
    x_val = x_train[:validation_splitpoint]
    x_train = x_train[validation_splitpoint:]
    y_val = y_train[:validation_splitpoint]
    y_train = y_train[validation_splitpoint:]

    history = model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val))

    loss, acc = model.evaluate(x_test, y_test)

    NeuralNetworkHistoryPlot.plot_accuracy(history, 'Shallow NN Tanh')
    NeuralNetworkHistoryPlot.plot_loss(history, 'Shallow NN Tanh')

    y_pred = model.predict(x_test).tolist()
    confusion_matrix = NeuralNetworkHistoryPlot.get_confusion_matrix(y_pred, y_test)
    home_diff, draw_diff, away_diff = NeuralNetworkHistoryPlot.get_percentage_differences(y_pred, y_test)
    print('Confusion Matrix:')
    print(confusion_matrix)
    print('Diffs')
    print(home_diff, draw_diff, away_diff)

    return acc


def shallow_neural_network_sigmoid(x_train, x_test, y_train, y_test, number_of_features):
    """
    function that trains a neural network with one hidden layer and sigmoid activation. The function prints the
    confusion matrix for the model and the training plot over its epochs
    :param x_train: input training data
    :param x_test: input test data
    :param y_train: output test data
    :param y_test: output test data
    :param number_of_features: number of input features to the network. Equal to the number of neurins in the first
    layer
    :return: the accuracy of the model
    """

    model = keras.Sequential([
        keras.Input(shape=(number_of_features,)),
        keras.layers.Dense(10, activation='sigmoid'),
        keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='sgd', loss=keras.losses.CategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

    validation_splitpoint = int(len(x_test) * 0.1)
    x_val = x_train[:validation_splitpoint]
    x_train = x_train[validation_splitpoint:]
    y_val = y_train[:validation_splitpoint]
    y_train = y_train[validation_splitpoint:]

    history = model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val))

    loss, acc = model.evaluate(x_test, y_test)

    NeuralNetworkHistoryPlot.plot_accuracy(history, 'Shallow NN Sigmoid')
    NeuralNetworkHistoryPlot.plot_loss(history, 'Shallow NN Sigmoid')

    y_pred = model.predict(x_test).tolist()
    confusion_matrix = NeuralNetworkHistoryPlot.get_confusion_matrix(y_pred, y_test)
    home_diff, draw_diff, away_diff = NeuralNetworkHistoryPlot.get_percentage_differences(y_pred, y_test)
    print('Confusion Matrix:')
    print(confusion_matrix)
    print('Diffs')
    print(home_diff, draw_diff, away_diff)

    return acc
