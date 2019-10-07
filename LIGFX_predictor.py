import sys
from LIGFX import *
from sklearn.svm import SVC


def prepare_dataset(input_data_filename):
    ligfx_analysis = LIGFX(input_data_filename)
    ligfx_analysis.holdout()
    return ligfx_analysis


def main(input_data_filename):
    ligfx_analysis = prepare_dataset(input_data_filename)

    #   Logistic Regression
    ligfx_analysis.create_classifier()
    ligfx_analysis.run_default_analysis()
    ligfx_analysis.print_performance("LR")

    #   SVM
    ligfx_analysis.create_classifier(SVC(kernel='linear'))
    ligfx_analysis.run_default_analysis()
    ligfx_analysis.print_performance("SVM")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: %s <input data filename>" % sys.argv[0])
        sys.exit()
    main(sys.argv[1])
