import os
import argparse
from LIGFX import *
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering


def prepare_dataset(input_data_filename):
    ligfx_analysis = LIGFX(input_data_filename)
    ligfx_analysis.holdout()
    return ligfx_analysis


def summary_statistics(ligfx_analysis, folder_path, out_prefix):
    ligfx_statistics = LIGFXStatistics(ligfx_analysis)
    summary_outname = folder_path + out_prefix + "_summary.csv"
    correlation_outname = folder_path + out_prefix + "_correlation.csv"
    ligfx_statistics.write_summary_stat_matrix()
    ligfx_statistics.write_correlation_matrix()
    ligfx_statistics.write_correlation_matrix_csv(correlation_outname)
    ligfx_statistics.write_summary_stat_csv(summary_outname)


def exploratory_data_analysis(ligfx_analysis, folder_path, out_prefix):
    ligfx_analysis.run_pca()
    pca_loadings = folder_path + out_prefix + "_pca_1st.dat"
    pca_filename = folder_path + out_prefix + "_pca_scores.csv"
    ligfx_analysis.pca_analysis.write_loadings(0, pca_loadings)
    ligfx_analysis.pca_analysis.write_n_selected_components()
    ligfx_analysis.pca_analysis.write_pca_results()
    ligfx_analysis.pca_analysis.write_scores(pca_filename)
    reduced_ligfx_analysis = ligfx_analysis.pca_analysis.create_reduced_dataset()
    return reduced_ligfx_analysis


def prediction(ligfx_analysis, folder_path, out_prefix):
    classifier_dict = {
       'LR': LogisticRegression(solver="lbfgs"),
       'SVM': SVC(kernel='linear'),
       'RF': RandomForestClassifier(n_estimators=1000)
    }
    featureimportance_filename = folder_path + out_prefix + "_" + "Fimportances_"
    crossvalidation_filename = folder_path + out_prefix + "_" + "crossvalidation_"
    prediction_filename = folder_path + out_prefix + "_" + "cv_predictions.dat"
    outfile = open(prediction_filename, "w")
    outfile.write("%5s %5s %5s\n" % ('LR', 'SVM', 'RF'))
    predictions = {}
    stdout.write("LIGFX:------------PREDICTIONS-------------\n")
    for (classifier_name, classifier_method) in classifier_dict.items():
        ligfx_analysis.create_classifier(classifier_method, classifier_name)
        ligfx_analysis.run_default_analysis()
        predictions[classifier_name] = ligfx_analysis.run_cross_validation()
        ligfx_analysis.cross_validation_performance.write_performance_csv(crossvalidation_filename +
                                                                          classifier_name + ".csv")
        if classifier_name == "RF":
            print_coefficients(featureimportance_filename, classifier_name,
                               ligfx_analysis.classifier.feature_importances_)
        else:
            print_coefficients(featureimportance_filename, classifier_name, ligfx_analysis.classifier.coef_[0])
    for lr, svm, rf in zip(predictions['LR'], predictions['SVM'], predictions['RF']):
        outfile.write("%5s %5s %5s\n" % (lr, svm, rf))
    outfile.close()
    

def cluster_analysis(ligfx_analysis, folder_path, out_prefix):
    cluster_dict = {
        'KMeans': KMeans(n_clusters=2, random_state=0),
        'Hierarchical': AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
    }
    cluster_filename = folder_path + out_prefix + "_Cluster.dat"
    stdout.write("LIGFX:---------------CLUSTER--ANALYSIS---------------\n")
    for (name, method) in cluster_dict.items():
        ligfx_cluster_analysis = LIGFXCluster(ligfx_analysis)
        ligfx_cluster_analysis.run_cluster_analysis(method, name)
        ligfx_cluster_analysis.write_cluster_purity(class_labels=["Inh", "Act"])


def create_folder(out_prefix):
    folder_path = os.getcwd() + "/" + "Outputs" + "_" + out_prefix + "/"
    if os.path.isdir(folder_path):
        pass
    else:
        os.mkdir(folder_path)
    return folder_path


def parsing_options():
    options = argparse.Namespace()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', action='store', required=True, dest='Input', help='input datafile')
    parser.add_argument('-p', action='store', dest='Prefix', default="1", help=' output prefix')
    parser.parse_args(namespace=options)
    input_data_filename = options.Input
    out_prefix = options.Prefix
    return input_data_filename, out_prefix


def main():
    input_data_filename, out_prefix = parsing_options()
    ligfx_analysis = prepare_dataset(input_data_filename)

    folder_path = create_folder(out_prefix)

    summary_statistics(ligfx_analysis, folder_path, out_prefix)
    cluster_analysis(ligfx_analysis, folder_path, out_prefix)
    prediction(ligfx_analysis, folder_path, out_prefix)
    reduced_ligfx_analysis = exploratory_data_analysis(ligfx_analysis, folder_path, out_prefix)
    reduced_ligfx_analysis.holdout()
    prediction(reduced_ligfx_analysis, folder_path, "reduced")


def print_coefficients(filename, name, vector):
    importance_filename = open(filename + name + '.dat', 'w')
    for variable_weight in vector:
        importance_filename.write('%.4f\n' % variable_weight)
    importance_filename.close()


if __name__ == '__main__':
    # if len(sys.argv) != 2:
    #   print("Usage: %s <input data filename>" % sys.argv[0])
    #   sys.exit()
    main()
