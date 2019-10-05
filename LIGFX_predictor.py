import sys
from LIGFX import *
from sklearn.svm import SVC


def prepare_dataset(input_data_filename):
    ligfx_analysis = LIGFX(input_data_filename)
    ligfx_analysis.holdout()
    return ligfx_analysis


def run_full_analysis(ligfx_analysis):
    ligfx_analysis.run_training()
    ligfx_analysis.run_prediction()
    ligfx_analysis.get_confusion_matrix()


def main(input_data_filename):
    ligfx_analysis = prepare_dataset(input_data_filename)

    #   Logistic Regression
    ligfx_analysis.create_classifier()
    run_full_analysis(ligfx_analysis)
    print("\nLR:")
    print(ligfx_analysis.conf_mat)

    #   SVM
    ligfx_analysis.create_classifier(SVC(kernel='linear'))
    run_full_analysis(ligfx_analysis)
    print("\nSVM:")
    print(ligfx_analysis.conf_mat)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: %s <input data filename>" % sys.argv[0])
        sys.exit()
    main(sys.argv[1])
