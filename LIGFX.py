from sys import stdout
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, \
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score

performance_measures = {
    'accuracy': accuracy_score,
    'balanced_accuracy': balanced_accuracy_score,
    'precision': precision_score,
    'recall': recall_score,
    'f1': f1_score
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


class LIGFXCrossValPerformance:
    def __init__(self, ligfx_object, n_fold=10):
        self.LIGFX = ligfx_object
        self.performance_table = self.cross_validation(n_fold)

    def cross_validation(self, n_fold=10):
        scoring = list(performance_measures.keys())

        cv_scores = cross_validate(self.LIGFX.classifier, self.LIGFX.input_x, self.LIGFX.input_y,
                                   cv=n_fold, scoring=scoring, return_train_score=True)
        cv_data = pd.DataFrame.from_dict(cv_scores)
        return cv_data

    def write_performance(self, output_filename=None):
        file_handle = open(output_filename, 'a') if output_filename else stdout

        t_performance_table = self.performance_table.transpose()

        file_handle.write('LIGFX:-------------------------------------------------\n')
        file_handle.write('LIGFX: model                  %s\n' % self.LIGFX.classifier_name)
        file_handle.write('LIGFX: cross validation\n')
        for i in range(t_performance_table.shape[0]):
            file_handle.write('LIGFX: %-25s ' % t_performance_table.index[i])
            for j in range(t_performance_table.shape[1]):
                file_handle.write("%5.2f" % t_performance_table.iloc[i, j])
            file_handle.write("\n")

        if file_handle is not stdout:
            file_handle.close()

    def write_performance_csv(self, output_filename=None):
        if output_filename is not None:
            self.performance_table.to_csv(output_filename)

class LIGFXPca:
    def __init__(self,ligfx_object,n_components):
        self.LIGFX = ligfx_object
        self.pcadata,self.variance,self.components=self.principal_component(n_components)

    def principal_component(self,n_components):
        pca = PCA(n_components= n_components, svd_solver='arpack')
        components=pca.fit_transform(self.LIGFX.input_x)
        list_col= [ 'principal component %d' % i for i in range(1,n_components + 1)]
        principalDf = pd.DataFrame(data=components, columns=list_col)
        return principalDf,pca.explained_variance_ratio_,pca.components_

    def write_pca(self,output_filename=None):
        file_handle = open(output_filename, 'a') if output_filename else stdout

        file_handle.write('LIGFX:-------------------------------------------------\n')
        file_handle.write('LIGFX: PCA variance\n')
        print (self.variance)
        print ('Somma varianze: %lf' % self.variance.sum())

    def write_contribution(self,component=0,output_filename=None):
        file_handle = open(output_filename, 'a') if output_filename else stdout
        file_handle.write("Feature      -  Contribution to component %d\n" % (component + 1))
        for ind in np.argsort(self.components[component])[::-1]:
            file_handle.write("%s               %lf \n" % (self.LIGFX.names[ind], self.components[0][ind]))

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
        self.cross_validation_performance = None
        self.names= list(self.input_x.columns.values)

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

    def run_default_analysis(self, output_filename=None):
        self.run_training()
        self.run_prediction()
        self.performance.write_performance(output_filename)

    def run_cross_validation(self, n_fold=10, output_filename=None):
        self.cross_validation_performance = LIGFXCrossValPerformance(self, n_fold)
        self.cross_validation_performance.write_performance(output_filename)

    def run_pca(self,n_components=3,output_filename=None):
        self.pca_analysis = LIGFXPca(self,n_components)
        self.pca_analysis.write_pca(output_filename)
        return self.pca_analysis

    def run_pca_curve(self,output_filename=None):
        file_handle = open(output_filename, 'a') if output_filename else stdout
        file_handle.write('LIGFX: PCA variance\n')
        file_handle.write    ('N_components  -  Total Variance\n')
        for i in range(2,40):
            self.pca_analysis = LIGFXPca(self, i)
            file_handle.write(' %d            %lf\n' % (i,self.pca_analysis.variance.sum()))
