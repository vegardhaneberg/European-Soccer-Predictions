from Preprocessing import GetData
from Preprocessing import PCA

from Models import TwoLayerNeuralNetwork
from Models import SVM
from Models import DeepNeuralNetworks
from Models import ShallowNeuralNetworks

from Plots import ModelAccuracyPlot

from sklearn.model_selection import train_test_split
import random


def optimal_pca_components():
    """
    function that finds the optimal number of components in the pca analysis
    """

    data = GetData.load_preprocessed_data('../Data/processed_data_10_features.csv')
    target_feature = 'match_result'

    accuracies = []

    for components in range(1, 6):
        pca_df = PCA.pca(data, target_feature, components)

        list_data = GetData.convert_df_to_lists(pca_df)

        neural_net = TwoLayerNeuralNetwork.neural_network(list_data, components)

        accuracies.append(neural_net)

    print(accuracies)


def run_all_models(input_vector, output_vector):
    random_state = random.randint(1, 10)
    print(random_state)
    x_train, x_test, y_train, y_test = train_test_split(input_vector, output_vector, test_size=0.25, random_state=random_state)

    # Creating and training the models without pca
    print('Creating and training SVM')
    svm_acc = SVM.support_vector_machine(x_train, x_test, y_train, y_test)

    y_train = GetData.convert_output_vector_to_nn_format(y_train)
    y_test = GetData.convert_output_vector_to_nn_format(y_test)
    print('Creating and training neural networks')
    neural_net_acc = TwoLayerNeuralNetwork.neural_network(x_train, x_test, y_train, y_test, 13)
    neural_net_sigmoid_acc = TwoLayerNeuralNetwork.neural_network_sigmoid(x_train, x_test, y_train, y_test, 13)
    neural_net_huge_acc = TwoLayerNeuralNetwork.huge_neural_network(x_train, x_test, y_train, y_test, 13)
    neural_net_shallow_acc = TwoLayerNeuralNetwork.shallow_neural_network(x_train, x_test, y_train, y_test, 13)

    # Creating and training the models with pca
    df = GetData.create_df_from_two_lists([input_vector, output_vector])

    pca_df = PCA.pca(df, 'match_result', 2)

    list_data = GetData.convert_df_to_lists(pca_df, False)
    x_train, x_test, y_train, y_test = train_test_split(list_data[0], list_data[1], test_size=0.25, random_state=random_state)

    print('Creating and training SVM with PCA')
    svm_pca_acc = SVM.support_vector_machine(x_train, x_test, y_train, y_test)

    y_train = GetData.convert_output_vector_to_nn_format(y_train)
    y_test = GetData.convert_output_vector_to_nn_format(y_test)
    print('Creating and training neural network with PCA')
    neural_net_pca_acc = TwoLayerNeuralNetwork.neural_network(x_train, x_test, y_train, y_test, 2)

    # Presenting the results
    models = ['Neural Network', 'Neural Network sigmoid', 'Neural Network huge', 'Neural Network shallow',
              'Support Vector Machine', 'SVM PCA', 'Neural Network PCA']
    accuracies = [neural_net_acc, neural_net_sigmoid_acc, neural_net_huge_acc, neural_net_shallow_acc,
                  svm_acc, svm_pca_acc, neural_net_pca_acc]
    ModelAccuracyPlot.present_results(models, accuracies)


def run_all_models_from_preprocessed_data(path):
    data = GetData.load_preprocessed_data(path)
    data = data.drop(['avg_home_win_odds', 'avg_draw_odds', 'avg_away_win_odds'], axis=1)

    input_vector, output_vector = GetData.convert_df_to_lists(data, False)

    random_state = random.randint(1, 10)

    #run_all_shallow(input_vector, output_vector, random_state)
    #run_all_two_layer(input_vector, output_vector, random_state)
    run_all_deep(input_vector, output_vector, random_state)


def run_all_shallow(input_vector, output_vector, random_state):
    """
    function that runs all shallow neural networks, both with and without pca and prints the results
    :param input_vector: a list with lists that contains the input to the models
    :param output_vector: a list that contains the match results on the format 0, 1 or 2
    :param random_state: the state for the train test split
    """

    x_train, x_test, y_train, y_test = train_test_split(input_vector, output_vector, test_size=0.25, random_state=random_state)
    y_train = GetData.convert_output_vector_to_nn_format(y_train)
    y_test = GetData.convert_output_vector_to_nn_format(y_test)

    shallow_nn_relu_acc = ShallowNeuralNetworks.shallow_neural_network_relu(x_train, x_test, y_train, y_test, 10)
    shallow_nn_tanh_acc = ShallowNeuralNetworks.shallow_neural_network_tanh(x_train, x_test, y_train, y_test, 10)
    shallow_nn_sigmoid_acc = ShallowNeuralNetworks.shallow_neural_network_sigmoid(x_train, x_test, y_train, y_test, 10)

    df = GetData.create_df_from_two_lists([input_vector, output_vector])

    pca_df = PCA.pca(df, 'match_result', 2)

    list_data = GetData.convert_df_to_lists(pca_df, False)
    x_train, x_test, y_train, y_test = train_test_split(list_data[0], list_data[1], test_size=0.25, random_state=1)
    y_train = GetData.convert_output_vector_to_nn_format(y_train)
    y_test = GetData.convert_output_vector_to_nn_format(y_test)

    shallow_nn_relu_pca_acc = ShallowNeuralNetworks.shallow_neural_network_relu(x_train, x_test, y_train, y_test, 2)
    shallow_nn_tanh_pca_acc = ShallowNeuralNetworks.shallow_neural_network_tanh(x_train, x_test, y_train, y_test, 2)
    shallow_nn_sigmoid_pca_acc = ShallowNeuralNetworks.shallow_neural_network_sigmoid(x_train, x_test, y_train, y_test, 2)

    model_names = ['Shallow NN With relu', 'Shallow NN With tanh', 'Shallow NN With Sigmoid',
                   'Shallow NN With relu and PCA', 'Shallow NN With tanh and PCA', 'Shallow NN With Sigmoid and PCA']
    accuracies = [shallow_nn_relu_acc, shallow_nn_tanh_acc, shallow_nn_sigmoid_acc, shallow_nn_relu_pca_acc,
                  shallow_nn_tanh_pca_acc, shallow_nn_sigmoid_pca_acc]

    ModelAccuracyPlot.present_results(model_names, accuracies)


def run_all_two_layer(input_vector, output_vector, random_state):
    """
    function that runs all neural networks with two hidden layers, both with and without pca and prints the results
    :param input_vector: a list with lists that contains the input to the models
    :param output_vector: a list that contains the match results on the format 0, 1 or 2
    :param random_state: the state for the train test split
    """
    x_train, x_test, y_train, y_test = train_test_split(input_vector, output_vector, test_size=0.25, random_state=random_state)
    y_train = GetData.convert_output_vector_to_nn_format(y_train)
    y_test = GetData.convert_output_vector_to_nn_format(y_test)

    two_layer_nn_relu_acc = TwoLayerNeuralNetwork.two_layer_neural_network_relu(x_train, x_test, y_train, y_test, 10)
    two_layer_nn_tanh_acc = TwoLayerNeuralNetwork.two_layer_neural_network_tanh(x_train, x_test, y_train, y_test, 10)
    two_layer_nn_sigmoid_acc = TwoLayerNeuralNetwork.two_layer_neural_network_sigmoid(x_train, x_test, y_train, y_test, 10)

    df = GetData.create_df_from_two_lists([input_vector, output_vector])

    pca_df = PCA.pca(df, 'match_result', 2)

    list_data = GetData.convert_df_to_lists(pca_df, False)
    x_train, x_test, y_train, y_test = train_test_split(list_data[0], list_data[1], test_size=0.25, random_state=1)
    y_train = GetData.convert_output_vector_to_nn_format(y_train)
    y_test = GetData.convert_output_vector_to_nn_format(y_test)

    two_layer_nn_relu_pca_acc = TwoLayerNeuralNetwork.two_layer_neural_network_relu(x_train, x_test, y_train, y_test, 2)
    two_layer_nn_tanh_pca_acc = TwoLayerNeuralNetwork.two_layer_neural_network_tanh(x_train, x_test, y_train, y_test, 2)
    two_layer_nn_sigmoid_pca_acc = TwoLayerNeuralNetwork.two_layer_neural_network_sigmoid(x_train, x_test, y_train, y_test, 2)

    model_names = ['Two Hidden Layer NN With relu', 'Two Hidden Layer NN With tanh', 'Two Hidden Layer NN With Sigmoid',
                   'Two Hidden Layer NN With relu and PCA', 'Two Hidden Layer NN With tanh and PCA',
                   'Two Hidden Layer NN With Sigmoid and PCA']
    accuracies = [two_layer_nn_relu_acc, two_layer_nn_tanh_acc, two_layer_nn_sigmoid_acc,
                  two_layer_nn_relu_pca_acc, two_layer_nn_tanh_pca_acc, two_layer_nn_sigmoid_pca_acc]

    ModelAccuracyPlot.present_results(model_names, accuracies)


def run_all_deep(input_vector, output_vector, random_state):
    """
    function that runs all deep neural networks, both with and without pca and prints the results
    :param input_vector: a list with lists that contains the input to the models
    :param output_vector: a list that contains the match results on the format 0, 1 or 2
    :param random_state: the state for the train test split
    """

    x_train, x_test, y_train, y_test = train_test_split(input_vector, output_vector, test_size=0.25, random_state=random_state)
    y_train = GetData.convert_output_vector_to_nn_format(y_train)
    y_test = GetData.convert_output_vector_to_nn_format(y_test)

    deep_nn_relu_acc = DeepNeuralNetworks.deep_neural_network_relu(x_train, x_test, y_train, y_test, 10)
    deep_nn_tanh_acc = DeepNeuralNetworks.deep_neural_network_tanh(x_train, x_test, y_train, y_test, 10)
    deep_nn_sigmoid_acc = DeepNeuralNetworks.deep_neural_network_sigmoid(x_train, x_test, y_train, y_test, 10)

    df = GetData.create_df_from_two_lists([input_vector, output_vector])

    pca_df = PCA.pca(df, 'match_result', 2)

    list_data = GetData.convert_df_to_lists(pca_df, False)
    x_train, x_test, y_train, y_test = train_test_split(list_data[0], list_data[1], test_size=0.25, random_state=1)
    y_train = GetData.convert_output_vector_to_nn_format(y_train)
    y_test = GetData.convert_output_vector_to_nn_format(y_test)

    deep_nn_relu_pca_acc = DeepNeuralNetworks.deep_neural_network_relu(x_train, x_test, y_train, y_test, 2)
    deep_nn_tanh_pca_acc = DeepNeuralNetworks.deep_neural_network_tanh(x_train, x_test, y_train, y_test, 2)
    deep_nn_sigmoid_pca_acc = DeepNeuralNetworks.deep_neural_network_sigmoid(x_train, x_test, y_train, y_test, 2)

    model_names = ['Deep NN With relu', 'Deep NN With tanh', 'Deep NN With Sigmoid',
                   'Deep NN With relu and PCA', 'Deep NN With tanh and PCA', 'Deep NN With Sigmoid and PCA']
    accuracies = [deep_nn_relu_acc, deep_nn_tanh_acc, deep_nn_sigmoid_acc,
                  deep_nn_relu_pca_acc, deep_nn_tanh_pca_acc, deep_nn_sigmoid_pca_acc]

    ModelAccuracyPlot.present_results(model_names, accuracies)


def run_best_ann_and_svm(data):
    """
    function that runs the best svm and neural network (the shallow neural network with the relu activation function)
    and prints their confusion matrices and accuracies. The difference in probabilities for the neural network is also
    printed.
    :param data: dataframe containing the preprocessed data
    """
    state = random.randint(1, 10)
    #data = GetData.load_preprocessed_data(path)
    data = data.drop(['avg_home_win_odds', 'avg_draw_odds', 'avg_away_win_odds'], axis=1)

    input_vector, output_vector = GetData.convert_df_to_lists(data, False)

    x_train, x_test, y_train, y_test = train_test_split(input_vector, output_vector, test_size=0.25, random_state=state)

    print('SVM running...')
    svm_acc = SVM.support_vector_machine(x_train, x_test, y_train, y_test, 10 ** 7, 10 ** -4)

    y_train = GetData.convert_output_vector_to_nn_format(y_train)
    y_test = GetData.convert_output_vector_to_nn_format(y_test)

    print('Neural Network running...')
    print('The confusion matrix and probability differences will be printed after the NN has trained')
    shallow_nn_relu_acc = ShallowNeuralNetworks.shallow_neural_network_relu(x_train, x_test, y_train, y_test, 10)

    model_names = ['SVM', 'Shallow NN With relu']
    accuracies = [svm_acc, shallow_nn_relu_acc]

    ModelAccuracyPlot.present_results(model_names, accuracies)


def std_best_models(path):
    """
    function that runs the best three neural networks five times and print their accuracy
    :param path: path to the preprocessed data. If the provided preprocessed data is used:
    Data/preprocessed_data.csv
    """
    data = GetData.load_preprocessed_data(path)
    data = data.drop(['avg_home_win_odds', 'avg_draw_odds', 'avg_away_win_odds'], axis=1)

    input_vector, output_vector = GetData.convert_df_to_lists(data, False)

    shallow_tanh_accs = []
    deep_relu_accs = []
    deep_tanh_accs = []

    for i in range(5):
        random_state = random.randint(1, 10)
        x_train, x_test, y_train, y_test = train_test_split(input_vector, output_vector, test_size=0.25, random_state=random_state)

        y_train = GetData.convert_output_vector_to_nn_format(y_train)
        y_test = GetData.convert_output_vector_to_nn_format(y_test)

        shallow_nn_relu_acc = ShallowNeuralNetworks.shallow_neural_network_relu(x_train, x_test, y_train, y_test, 10)
        deep_nn_relu_acc = DeepNeuralNetworks.deep_neural_network_relu(x_train, x_test, y_train, y_test, 10)
        deep_nn_tanh_acc = DeepNeuralNetworks.deep_neural_network_tanh(x_train, x_test, y_train, y_test, 10)

        shallow_tanh_accs.append(shallow_nn_relu_acc)
        deep_relu_accs.append(deep_nn_relu_acc)
        deep_tanh_accs.append(deep_nn_tanh_acc)

    print('Two layer:', shallow_tanh_accs)
    print('Deep relu:', deep_relu_accs)
    print('Deep tanh:', deep_tanh_accs)


def std_svm(path):
    """
    function that runs the best svm five times and print their accuracy
    :param path: path to the preprocessed data. If the provided preprocessed data is used:
    Data/preprocessed_data.csv
    """
    data = GetData.load_preprocessed_data(path)
    data = data.drop(['avg_home_win_odds', 'avg_draw_odds', 'avg_away_win_odds'], axis=1)

    input_vector, output_vector = GetData.convert_df_to_lists(data, False)

    svm_accs = []
    for i in range(5):
        random_state = random.randint(1, 10)
        x_train, x_test, y_train, y_test = train_test_split(input_vector, output_vector, test_size=0.25, random_state=random_state)

        print('SVM running...')
        svm_acc = SVM.support_vector_machine(x_train, x_test, y_train, y_test, 10 ** 7, 10 ** -4)
        svm_accs.append(svm_acc)

    print('svm accuracies:', svm_accs)


def run_best_ann_and_svm_from_raw_data():
    """
    function that runs the best neural network and svm without preprocessed data
    """
    data = GetData.preprocessing()
    run_best_ann_and_svm(data)


def run_best_ann_and_svm_from_preprocessed_data():
    """
    function that runs the best neural network and svm from preprocessed data
    """
    data = GetData.load_preprocessed_data()
    run_best_ann_and_svm(data)


def main():
    path = 'Enter/Your/Path/Here'
    #run_best_ann_and_svm_from_preprocessed_data()
    run_best_ann_and_svm_from_raw_data()


if __name__ == '__main__':
    main()

