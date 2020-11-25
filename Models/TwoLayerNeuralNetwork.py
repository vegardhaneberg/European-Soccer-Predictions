from tensorflow import keras
from Preprocessing import GetData
from Plots import NeuralNetworkHistoryPlot

from sklearn.model_selection import train_test_split


def two_layer_neural_network_relu(x_train, x_test, y_train, y_test, number_of_features):
    """
    function that trains a neural network with two hidden layers and relu activation. The function prints the
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

    NeuralNetworkHistoryPlot.plot_accuracy(history, 'Two Hidden Layers NN Relu')
    NeuralNetworkHistoryPlot.plot_loss(history, 'Two Hidden Layers NN Relu')

    y_pred = model.predict(x_test).tolist()
    confusion_matrix = NeuralNetworkHistoryPlot.get_confusion_matrix(y_pred, y_test)
    home_diff, draw_diff, away_diff = NeuralNetworkHistoryPlot.get_percentage_differences(y_pred, y_test)
    print(confusion_matrix)
    print(home_diff, draw_diff, away_diff)

    return acc


def two_layer_neural_network_sigmoid(x_train, x_test, y_train, y_test, number_of_features):
    """
    function that trains a neural network with two hidden layers and sigmoid activation. The function prints the
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

    NeuralNetworkHistoryPlot.plot_accuracy(history, 'Two Hidden Layers NN Sigmoid')
    NeuralNetworkHistoryPlot.plot_loss(history, 'Two Hidden Layers NN Sigmoid')

    y_pred = model.predict(x_test).tolist()
    confusion_matrix = NeuralNetworkHistoryPlot.get_confusion_matrix(y_pred, y_test)
    home_diff, draw_diff, away_diff = NeuralNetworkHistoryPlot.get_percentage_differences(y_pred, y_test)
    print(confusion_matrix)
    print(home_diff, draw_diff, away_diff)

    return acc


def two_layer_neural_network_tanh(x_train, x_test, y_train, y_test, number_of_features):
    """
    function that trains a neural network with two hidden layers and tanh activation. The function prints the
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

    NeuralNetworkHistoryPlot.plot_accuracy(history, 'Two Hidden Layers NN Tanh')
    NeuralNetworkHistoryPlot.plot_loss(history, 'Two Hidden Layers NN Tanh')

    y_pred = model.predict(x_test).tolist()
    confusion_matrix = NeuralNetworkHistoryPlot.get_confusion_matrix(y_pred, y_test)
    home_diff, draw_diff, away_diff = NeuralNetworkHistoryPlot.get_percentage_differences(y_pred, y_test)
    print('Confusion Matrix:')
    print(confusion_matrix)
    print('Diffs')
    print(home_diff, draw_diff, away_diff)

    return acc


def main():
    data = GetData.load_preprocessed_data('/Users/vegardhaneberg/PycharmProjects/SoccerGroupProject/Data/preprocessed_data.csv')
    data = data.drop(['avg_home_win_odds', 'avg_draw_odds', 'avg_away_win_odds'], axis=1)

    input_vector, output_vector = GetData.convert_df_to_lists(data, False)

    x_train, x_test, y_train, y_test = train_test_split(input_vector, output_vector, test_size=0.25, random_state=1)
    y_train = GetData.convert_output_vector_to_nn_format(y_train)
    y_test = GetData.convert_output_vector_to_nn_format(y_test)
    two_layer_neural_network_tanh(x_train, x_test, y_train, y_test, 10)


if __name__ == '__main__':
    main()
