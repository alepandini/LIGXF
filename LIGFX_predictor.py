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

    #   Logistic Regression
    ligfx_analysis.create_classifier()
    ligfx_analysis.run_default_analysis()
    ligfx_analysis.write_performance()

    #   SVM
    ligfx_analysis.create_classifier(SVC(kernel='linear'), 'SVM')
    ligfx_analysis.run_default_analysis()
    ligfx_analysis.write_performance()

    #   RF
    ligfx_analysis.create_classifier(RandomForestClassifier(n_estimators=1000), 'RF')
    ligfx_analysis.run_default_analysis()
    ligfx_analysis.write_performance()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: %s <input data filename>" % sys.argv[0])
        sys.exit()
    main(sys.argv[1])
