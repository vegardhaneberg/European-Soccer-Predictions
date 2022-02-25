# TDT4173

This is a despription of how to run the code for the group project in TDT4173. A SVM and ANN will be trained to predict
the outcome of football matches!

The preprocessed data is stored in Data/preprocessed_data.csv. To run the models from this preprocessed data do the following:
  1. Clone the repo locally (navigate to your preferred directory and run:
  ```
     git clone https://github.com/vegardhaneberg/TDT4173.git)
  ```
  4. Run:
  ```
  pip install -r requirements.txt
  ```
  6. Run the main.py file
     This will run the best support vector machine and the best neural network on the preprocessed data. The confusion matrix
     for the svm will be printed before the neural network runs. After the neural network has finished training, the training
     plots will be plotted. Finally the NN-confusion matrix and the probability differences will be printed.

The models can also run from raw data. Then do the following:
  1. Clone the repo locally (navigate to your preferred directory and run ```git clone https://github.com/vegardhaneberg/TDT4173.git```)
  2. Download the dataset from https://www.kaggle.com/hugomathien/soccer and place the database in the Data directory. 
     The database should be called 'database.sqlite'
  3. Comment out line 246 in Main.py
  4. Uncomment line 247 in Main.py and run the file
     This will run the best support vector machine and the best neural network from raw data. Please note that this will take
     approximately one hour. The confusion matrix for the svm will be printed before the neural network runs. After the neural 
     network has finished training, the training plots will be plotted. Finally the NN-confusion matrix and the probability 
     differences will be printed.
  

To create the plots presented in section 2.2 in the report do the following:
  1. Clone the repo locally (navigate to your preferred directory and run ```git clone https://github.com/vegardhaneberg/TDT4173.git```)
  2. Download the dataset from https://www.kaggle.com/hugomathien/soccer and place the database in the Data directory. 
     The database should be called 'database.sqlite'
  3. Run the file Plots/PlotPreprocessedData.py


  
