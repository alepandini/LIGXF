from sys import stdout
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, \
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

performance_measures = {
    'accuracy': accuracy_score,
    'balanced accuracy': balanced_accuracy_score,
    'precision': precision_score,
    'recall': recall_score,
    'f1_score': f1_score,
    'Matthews corr coeff': matthews_corrcoef
}


class LIGFXPerformance:
    def __init__(self, ligfx_object):
        self.LIGFX = ligfx_object
        self.conf_mat = self.get_confusion_matrix()
        self.performance_scores = {
            k: self.get_performance_score(v)
            for (k, v) in performance_measures.items()
        }

    def get_confusion_matrix(self):
        conf_mat = confusion_matrix(self.LIGFX.test_y, self.LIGFX.predicted_y)
        return conf_mat

    def get_performance_score(self, performance_function):
        score = performance_function(self.LIGFX.test_y, self.LIGFX.predicted_y)
        return score

    def write_performance(self, output_filename=None):
        file_handle = open(output_filename, 'a') if output_filename else stdout

        file_handle.write('LIGFX:-------------------------------------------------\n')
        file_handle.write('LIGFX: model                  %s\n' % self.LIGFX.classifier_name)
        for (k, v) in self.performance_scores.items():
            file_handle.write('LIGFX: %-20s   %4.2f\n' % (k, v))
        file_handle.write('LIGFX: Confusion matrix\n')
        for i in range(self.conf_mat.shape[0]):
            file_handle.write('LIGFX:                     ')
            for j in range(self.conf_mat.shape[1]):
                file_handle.write('%5d ' % self.conf_mat[i, j])
            file_handle.write("\n")

        if file_handle is not stdout:
            file_handle.close()


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
        self.classifier_name = None
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

    def create_classifier(self, method=LogisticRegression(solver="lbfgs"), classifier_name='LR'):
        self.classifier = method
        self.classifier_name = classifier_name

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

    def write_performance(self, output_filename=None):
        if self.trained:
            self.performance.write_performance(output_filename)
        else:
            print("Warning: the model was not trained.")
            return 1
