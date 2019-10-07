import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


class LIGFXPerformance:
    def __init__(self, ligfx_object):
        self.LIGFX = ligfx_object
        self.conf_mat = self.get_confusion_matrix()
        self.accuracy = self.get_accuracy()
        self.F1_score = self.get_f1()

    def print_performance(self, model_name=""):
        if self.accuracy is None:
            print("Warning: prediction was not run.")
            return 1
        else:
            print('LIGFX:-------------------------------------------------')
            print('LIGFX:Model     %s' % model_name)
            print('LIGFX:Accuracy  %4.2f' % self.accuracy)
            print('LIGFX:F1-score  %4.2f' % self.F1_score)
            print('LIGFX:Confusion matrix')
            for i in range(self.conf_mat.shape[0]):
                print('LIGFX:       ', end='')
                for j in range(self.conf_mat.shape[1]):
                    print('%5d ' % self.conf_mat[i, j], end='')
                print()

    def get_confusion_matrix(self):
        conf_mat = confusion_matrix(self.LIGFX.test_y, self.LIGFX.predicted_y)
        return conf_mat

    def get_accuracy(self):
        accuracy = accuracy_score(self.LIGFX.test_y, self.LIGFX.predicted_y)
        return accuracy

    def get_f1(self):
        f1 = f1_score(self.LIGFX.test_y, self.LIGFX.predicted_y)
        return f1


class LIGFX:
    def __init__(self, input_data_filename, normalise=True):
        self.input_data = self.__read_input(input_data_filename)
        if normalise:
            self.norm_input_data = self.__normalise()
            self.input_x = self.norm_input_data.drop('y', axis=1)
            self.input_y = self.norm_input_data['y']
        else:
            self.input_x = self.input_data.drop('y', axis=1)
            self.input_y = self.input_data['y']
        self.training_x = None
        self.training_y = None
        self.test_x = None
        self.test_y = None
        self.classifier = None
        self.trained = False
        self.predicted_y = None
        self.performance = None

    @staticmethod
    def __read_input(input_data_filename):
        input_data = pd.read_csv(input_data_filename, header=0)
        return input_data

    def __normalise(self):
        norm_data = (self.input_data - self.input_data.min()) / (self.input_data.max() - self.input_data.min())
        return norm_data

    def holdout(self, test_percentage=30):
        x_training, x_test, y_training, y_test = train_test_split(
            self.input_x, self.input_y, test_size=(test_percentage/100), random_state=0)
        self.training_x = x_training
        self.training_y = y_training
        self.test_x = x_test
        self.test_y = y_test

    def create_classifier(self, method=LogisticRegression(solver="lbfgs")):
        self.classifier = method

    def run_training(self):
        if (self.training_x is None) or (self.training_y is None):
            print("Warning: training set not defined.")
            return 1
        if self.classifier is None:
            print("Warning: machine learning method not defined.")
            return 1
        self.classifier.fit(self.training_x, self.training_y)
        self.trained = True
        return 0

    def run_prediction(self):
        if self.trained:
            self.predicted_y = self.classifier.predict(self.test_x)
            self.performance = LIGFXPerformance(self)
        else:
            print("Warning: the model was not trained.")
            return 1

    def run_default_analysis(self):
        self.run_training()
        self.run_prediction()

    def print_performance(self, model_name=""):
        if self.trained:
            self.performance.print_performance(model_name)
        else:
            print("Warning: the model was not trained.")
            return 1
