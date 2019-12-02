import sys
from LIGFX import *
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering


def prepare_dataset(input_data_filename):
    ligfx_analysis = LIGFX(input_data_filename)
    ligfx_analysis.holdout()
    return ligfx_analysis


def summary_statistics(ligfx_analysis):
    ligfx_statistics = LIGFXStatistics(ligfx_analysis)
    ligfx_statistics.write_summary_stat_matrix()
    ligfx_statistics.write_correlation_matrix()


def exploratory_data_analysis(ligfx_analysis):
    ligfx_analysis.run_pca()
    ligfx_analysis.pca_analysis.write_loadings(0)
    ligfx_analysis.pca_analysis.write_n_selected_components()
    reduced_ligfx_analysis = ligfx_analysis.pca_analysis.create_reduced_dataset()
    return reduced_ligfx_analysis


def prediction(ligfx_analysis):
    classifier_dict = {
       'LR': LogisticRegression(solver="lbfgs"),
       'SVM': SVC(kernel='linear'),
       'RF': RandomForestClassifier(n_estimators=1000)
    }
    for (classifier_name, classifier_method) in classifier_dict.items():
        ligfx_analysis.create_classifier(classifier_method, classifier_name)
        ligfx_analysis.run_default_analysis()
        ligfx_analysis.run_cross_validation()
        # ligfx_analysis.cross_validation_performance.write_performance_csv("cross_val_%s.csv" % classifier_name)


def cluster_analysis(ligfx_analysis):
    cluster_dict = {
        'KMeans': KMeans(n_clusters=2, random_state=0),
        'Hierarchical': AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
    }
    for (name, method) in cluster_dict.items():
        ligfx_cluster_analysis = LIGFXCluster(ligfx_analysis)
        ligfx_cluster_analysis.run_cluster_analysis(method, name)
        ligfx_cluster_analysis.write_2_clusters_separation()


def main(input_data_filename):
    ligfx_analysis = prepare_dataset(input_data_filename)
    # summary_statistics(ligfx_analysis)
    cluster_analysis(ligfx_analysis)
    # prediction(ligfx_analysis)
    # reduced_ligfx_analysis = exploratory_data_analysis(ligfx_analysis)
    # reduced_ligfx_analysis.holdout()
    # prediction(reduced_ligfx_analysis)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: %s <input data filename>" % sys.argv[0])
        sys.exit()
    main(sys.argv[1])
