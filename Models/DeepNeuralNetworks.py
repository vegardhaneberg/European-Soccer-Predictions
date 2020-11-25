from tensorflow import keras
from Plots import NeuralNetworkHistoryPlot


def deep_neural_network_relu(x_train, x_test, y_train, y_test, number_of_features):
    """
    function that creates and trains a neural network with 15 hidden layers and 10 neurons in each hidden layer. The
    hidden layers uses relu as activation function. The function plots the traning and validation accuracy and loss
    over its training epochs. Furthermore the function prints the confusion matrix and the differences in probability
    with the match outcome.
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
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(10, activation='relu'),
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

    NeuralNetworkHistoryPlot.plot_accuracy(history, 'Deep NN Relu')
    NeuralNetworkHistoryPlot.plot_loss(history, 'Deep NN Relu')

    y_pred = model.predict(x_test).tolist()
    confusion_matrix = NeuralNetworkHistoryPlot.get_confusion_matrix(y_pred, y_test)
    home_diff, draw_diff, away_diff = NeuralNetworkHistoryPlot.get_percentage_differences(y_pred, y_test)
    print('Confusion Matrix:')
    print(confusion_matrix)
    print('Diffs')
    print(home_diff, draw_diff, away_diff)

    return acc


def deep_neural_network_tanh(x_train, x_test, y_train, y_test, number_of_features):
    """
    function that creates and trains a neural network with 15 hidden layers and 10 neurons in each hidden layer. The
    hidden layers uses tanh as activation function. The function plots the traning and validation accuracy and loss
    over its training epochs. Furthermore the function prints the confusion matrix and the differences in probability
    with the match outcome.
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
        keras.layers.Dense(10, activation='tanh'),
        keras.layers.Dense(10, activation='tanh'),
        keras.layers.Dense(10, activation='tanh'),
        keras.layers.Dense(10, activation='tanh'),
        keras.layers.Dense(10, activation='tanh'),
        keras.layers.Dense(10, activation='tanh'),
        keras.layers.Dense(10, activation='tanh'),
        keras.layers.Dense(10, activation='tanh'),
        keras.layers.Dense(10, activation='tanh'),
        keras.layers.Dense(10, activation='tanh'),
        keras.layers.Dense(10, activation='tanh'),
        keras.layers.Dense(10, activation='tanh'),
        keras.layers.Dense(10, activation='tanh'),
        keras.layers.Dense(10, activation='tanh'),
        keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

    validation_splitpoint = int(len(x_test) * 0.1)
    x_val = x_train[:validation_splitpoint]
    x_train = x_train[validation_splitpoint:]
    y_val = y_train[:validation_splitpoint]
    y_train = y_train[validation_splitpoint:]

    history = model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val))

    loss, acc = model.evaluate(x_test, y_test)

    NeuralNetworkHistoryPlot.plot_accuracy(history, 'Deep NN Tanh')
    NeuralNetworkHistoryPlot.plot_loss(history, 'Deep NN Tanh')

    y_pred = model.predict(x_test).tolist()
    confusion_matrix = NeuralNetworkHistoryPlot.get_confusion_matrix(y_pred, y_test)
    home_diff, draw_diff, away_diff = NeuralNetworkHistoryPlot.get_percentage_differences(y_pred, y_test)
    print('Confusion Matrix:')
    print(confusion_matrix)
    print('Diffs')
    print(home_diff, draw_diff, away_diff)

    return acc


def deep_neural_network_sigmoid(x_train, x_test, y_train, y_test, number_of_features):
    """
    function that creates and trains a neural network with 15 hidden layers and 10 neurons in each hidden layer. The
    hidden layers uses sigmoid as activation function. The function plots the traning and validation accuracy and loss
    over its training epochs. Furthermore the function prints the confusion matrix and the differences in probability
    with the match outcome.
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
        keras.layers.Dense(10, activation='sigmoid'),
        keras.layers.Dense(10, activation='sigmoid'),
        keras.layers.Dense(10, activation='sigmoid'),
        keras.layers.Dense(10, activation='sigmoid'),
        keras.layers.Dense(10, activation='sigmoid'),
        keras.layers.Dense(10, activation='sigmoid'),
        keras.layers.Dense(10, activation='sigmoid'),
        keras.layers.Dense(10, activation='sigmoid'),
        keras.layers.Dense(10, activation='sigmoid'),
        keras.layers.Dense(10, activation='sigmoid'),
        keras.layers.Dense(10, activation='sigmoid'),
        keras.layers.Dense(10, activation='sigmoid'),
        keras.layers.Dense(10, activation='sigmoid'),
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

    NeuralNetworkHistoryPlot.plot_accuracy(history, 'Deep NN Sigmoid')
    NeuralNetworkHistoryPlot.plot_loss(history, 'Deep NN Sigmoid')

    y_pred = model.predict(x_test).tolist()
    confusion_matrix = NeuralNetworkHistoryPlot.get_confusion_matrix(y_pred, y_test)
    home_diff, draw_diff, away_diff = NeuralNetworkHistoryPlot.get_percentage_differences(y_pred, y_test)
    print('Confusion Matrix:')
    print(confusion_matrix)
    print('Diffs')
    print(home_diff, draw_diff, away_diff)

    return acc
