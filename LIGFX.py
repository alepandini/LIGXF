import pandas as pd
import numpy as np
from sys import stdout
from scipy import stats
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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
        self.performance_table, self.predict = self.cross_validation(n_fold)

    def cross_validation(self, n_fold=10):
        scoring = list(performance_measures.keys())

        cv_scores = cross_validate(self.LIGFX.classifier, self.LIGFX.input_x, self.LIGFX.input_y,
                                   cv=n_fold, scoring=scoring, return_train_score=True)
        cv_predict = cross_val_predict(self.LIGFX.classifier, self.LIGFX.input_x, y=self.LIGFX.input_y, cv=n_fold)
        cv_data = pd.DataFrame.from_dict(cv_scores)
        return cv_data, cv_predict

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


class LIGFXStatistics:
    def __init__(self, ligfx_object):
        self.LIGFX = ligfx_object
        self.summary_stat = self.calculate_statistic()
        self.correlation_mat = self.calculate_correlation()

    def calculate_statistic(self):
        summary_stat = pd.concat(
            [self.LIGFX.input_x.min(),
             self.LIGFX.input_x.quantile([0.25]).transpose(),
             self.LIGFX.input_x.mean(),
             self.LIGFX.input_x.median(),
             self.LIGFX.input_x.quantile([0.75]).transpose(),
             self.LIGFX.input_x.max(),
             self.LIGFX.input_x.std()],
            axis=1)
        summary_stat.columns = ["min", "Q1", "mean", "median", "Q3", "max", "std"]
        return summary_stat

    def calculate_correlation(self):
        matrix = np.zeros([self.LIGFX.n_input_features, self.LIGFX.n_input_features], dtype=float)
        for i in range(self.LIGFX.n_input_features):
            for j in range(self.LIGFX.n_input_features):
                matrix[i, j] = stats.pearsonr(self.LIGFX.input_x.iloc[:, i], self.LIGFX.input_x.iloc[:, j])[0]
        return matrix

    def write_correlation_matrix(self, output_filename=None):
        file_handle = open(output_filename, 'a') if output_filename else stdout

        file_handle.write('LIGFX:-------------------------------------------------\n')
        file_handle.write('LIGFX: correlation matrix\n')
        file_handle.write('LIGFX:   legend:\n')
        for i in range(self.LIGFX.n_input_features):
            file_handle.write('LIGFX:      %2d %s\n' % ((i + 1), self.LIGFX.input_x.columns[i]))

        for i in range(self.LIGFX.n_input_features):
            file_handle.write("LIGFX:      %5d " % (i + 1))
            for j in range(self.LIGFX.n_input_features):
                file_handle.write("%5.2f " % self.correlation_mat[i, j])
            file_handle.write("\n")

        if file_handle is not stdout:
            file_handle.close()

    def write_summary_stat_matrix(self, output_filename=None):
        file_handle = open(output_filename, 'a') if output_filename else stdout

        file_handle.write('LIGFX:-------------------------------------------------\n')
        file_handle.write('LIGFX: summary stat\n')

        file_handle.write('LIGFX:                ')
        for j in range(self.summary_stat.shape[1]):
            file_handle.write("%8s" % self.summary_stat.columns[j])
        file_handle.write('\n')

        for i in range(self.summary_stat.shape[0]):
            file_handle.write('LIGFX: %15s' % self.summary_stat.index[i])
            for j in range(self.summary_stat.shape[1]):
                file_handle.write("%8.3f" % self.summary_stat.iloc[i, j])
            file_handle.write('\n')

        if file_handle is not stdout:
            file_handle.close()

    def write_correlation_matrix_csv(self, output_filename=None):
        if output_filename is not None:
            pd.DataFrame(self.correlation_mat).to_csv(output_filename)

    def write_summary_stat_csv(self, output_filename=None):
        if output_filename is not None:
            pd.DataFrame(self.summary_stat).to_csv(output_filename)


class LIGFXCluster:
    def __init__(self, ligfx_object):
        self.LIGFX = ligfx_object
        self.method = None
        self.name = None
        self.clusters = None
        self.confusion_matrix = None
        self.cluster_purity = None

    def run_cluster_analysis(self, method=KMeans(n_clusters=2, random_state=0), name='KMeans'):
        self.method = method
        self.name = name
        self.clusters = self.method.fit(self.LIGFX.input_x)

        unique_labels = self.LIGFX.input_y.unique()
        self.confusion_matrix = pd.DataFrame(data=np.zeros((self.clusters.n_clusters, len(unique_labels)), dtype=int))
        for i in range(self.clusters.n_clusters):
            for j in range(len(unique_labels)):
                self.confusion_matrix.iloc[i, j] = np.array(
                    self.LIGFX.input_y[self.clusters.labels_ == i] == unique_labels[j]).sum()
        self.calculate_cluster_purity()

    def calculate_cluster_purity(self, method=KMeans(n_clusters=2, random_state=0), name='KMeans'):
        if self.method is None:
            self.run_cluster_analysis(method=method, name=name)
        self.cluster_purity = sum(self.confusion_matrix.max(axis=1))

    def write_cluster_purity(self, output_filename=None, class_labels=None):
        file_handle = open(output_filename, 'a') if output_filename else stdout

        file_handle.write('LIGFX:-------------------------------------------------\n')
        file_handle.write('LIGFX: Method:  %s \n' % self.name)
        file_handle.write('LIGFX: n clusters: %d \n' % self.clusters.n_clusters)
        file_handle.write('LIGFX: Maximum separation obtained: %d over %d samples\n' % (self.cluster_purity,
                                                                                        len(self.LIGFX.input_y)))
        if class_labels is None:
            class_labels = self.LIGFX.input_y.unique()
        file_handle.write('LIGFX:               ')
        for i in range(len(class_labels)):
            file_handle.write('%6s' % class_labels[i])
        file_handle.write('\n')

        for i in range(self.confusion_matrix.shape[0]):
            file_handle.write('LIGFX: Cluster %3d  ' % i)
            for j in range(self.confusion_matrix.shape[1]):
                file_handle.write("%6d" % self.confusion_matrix.iloc[i, j])
            file_handle.write('\n')

        if file_handle is not stdout:
            file_handle.close()


class LIGFXPCAnalysis:
    def __init__(self, ligfx_object):
        self.LIGFX = ligfx_object
        self.pca_input_x = None
        self.pca_variance = None
        self.pca_cumulative_variance = None
        self.pca_components = None
        self.pca_n_essential_components = 0

    def calculate_components(self):
        n_features = self.LIGFX.n_input_features
        pca = PCA(n_components=n_features, svd_solver='full')
        scores = pca.fit_transform(self.LIGFX.input_x)
        column_names = ['PC%d' % i for i in range(1, n_features + 1)]
        scores_data_frame = pd.DataFrame(data=scores, columns=column_names)
        self.pca_input_x = scores_data_frame
        self.pca_variance = pca.explained_variance_ratio_
        self.pca_cumulative_variance = np.cumsum(self.pca_variance)
        self.pca_components = pca.components_

    def select_components(self, n_components=0, percentage_threshold=80):
        if self.pca_cumulative_variance is None:
            self.calculate_components()
        if n_components == 0:
            n_components = 1 + np.argmax(self.pca_cumulative_variance > (percentage_threshold / 100))
        self.pca_n_essential_components = n_components

    def extract_reduced_data(self, n_components=0, percentage_threshold=80):
        if self.pca_n_essential_components is 0:
            self.select_components(n_components, percentage_threshold)
        reduced_input_data = self.pca_input_x.iloc[:, :self.pca_n_essential_components].copy()
        return reduced_input_data

    def create_reduced_dataset(self, n_components=0, percentage_threshold=80):
        input_data = self.extract_reduced_data(n_components, percentage_threshold)
        input_data['y'] = self.LIGFX.input_y.copy()
        reduced_ligfx = LIGFX(input_data, input_from_file=False)
        return reduced_ligfx

    def write_pca_results(self, output_filename=None):
        if self.pca_variance is None:
            self.calculate_components()

        file_handle = open(output_filename, 'a') if output_filename else stdout

        file_handle.write('LIGFX:-------------------------------------------------\n')
        file_handle.write('LIGFX: PCA - Variance table\n')
        file_handle.write('LIGFX: PC  variance cumulative_variance\n')
        for i in range(len(self.pca_variance)):
            print("LIGFX: %-4d%-9.3f%-8.3f" % (i + 1, self.pca_variance[i], self.pca_cumulative_variance[i]))

        if file_handle is not stdout:
            file_handle.close()

    def write_loadings(self, component=0, output_filename=None):
        if self.pca_components is None:
            self.calculate_components()

        file_handle = open(output_filename, 'a') if output_filename else stdout

        file_handle.write('LIGFX:-------------------------------------------------\n')
        file_handle.write('LIGFX: PCA - Contribution to component %d\n' % (component + 1))
        file_handle.write('LIGFX: Feature              loading\n')
        for ind in np.argsort(self.pca_components[component])[::-1]:
            file_handle.write("LIGFX: %-20s%8.3f \n" % (self.LIGFX.names[ind], self.pca_components[0][ind]))

        if file_handle is not stdout:
            file_handle.close()

    def write_n_selected_components(self, output_filename=None, n_components=0, percentage_threshold=80):
        if self.pca_n_essential_components is 0:
            self.select_components(n_components, percentage_threshold)

        file_handle = open(output_filename, 'a') if output_filename else stdout

        file_handle.write('LIGFX:-------------------------------------------------\n')
        file_handle.write('LIGFX: PCA \n')
        file_handle.write('LIGFX: number of selected components: %d\n' % self.pca_n_essential_components)

        if file_handle is not stdout:
            file_handle.close()

    def write_scores(self, output_filename=None):
        if output_filename:
            self.pca_input_x.to_csv(output_filename)
        else:
            stdout.write("Error: No pca outfile selected!\n")


class LIGFX:
    def __init__(self, input_data, normalise=True, input_from_file=True):
        if input_from_file:
            self.input_data = self.__read_input(input_data)
        else:
            self.input_data = input_data
        if normalise:
            self.norm_input_data = self.__normalise()
            self.input_x = self.norm_input_data.drop('y', axis=1)
            self.input_y = self.norm_input_data['y']
        else:
            self.input_x = self.input_data.drop('y', axis=1)
            self.input_y = self.input_data['y']
        self.n_input_features = self.input_x.shape[1]
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
        self.names = list(self.input_x.columns.values)
        self.pca_analysis = None

    def __del__(self):
        
        self.input_data = None
        self.input_x = None
        self.input_y = None
        self.n_input_features = None
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
        self.names = None
        self.pca_analysis = None

    @staticmethod
    def __read_input(input_data_filename):
        input_data = pd.read_csv(input_data_filename, header=0)
        return input_data

    def __normalise(self):
        norm_data = (self.input_data - self.input_data.min()) / (self.input_data.max() - self.input_data.min())
        return norm_data

    def holdout(self, test_percentage=30, seed = 0, strat= None):
        x_training, x_test, y_training, y_test = train_test_split(
            self.input_x, self.input_y, test_size=(test_percentage/100), random_state=seed, stratify = strat)
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

    def run_default_analysis(self, output_filename=None, write= True):
        self.run_training()
        self.run_prediction()
        if write:
            self.performance.write_performance(output_filename)
        else:
            return self.performance.get_performance_score(performance_measures['balanced_accuracy'])

    def run_cross_validation(self, n_fold=10, output_filename=None):
        self.cross_validation_performance = LIGFXCrossValPerformance(self, n_fold)
        self.cross_validation_performance.write_performance(output_filename)
        prediction = np.array([predict == real for predict, real in
                               zip(self.cross_validation_performance.predict, self.input_y)])
        return prediction

    def run_pca(self, n_components=0):
        self.pca_analysis = LIGFXPCAnalysis(self)
        self.pca_analysis.calculate_components()
        self.pca_analysis.select_components(n_components)
