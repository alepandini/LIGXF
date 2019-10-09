import sys
from LIGFX import *
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def prepare_dataset(input_data_filename):
    ligfx_analysis = LIGFX(input_data_filename)
    ligfx_analysis.holdout()
    return ligfx_analysis


def main(input_data_filename):
    ligfx_analysis = prepare_dataset(input_data_filename)

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


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: %s <input data filename>" % sys.argv[0])
        sys.exit()
    main(sys.argv[1])
