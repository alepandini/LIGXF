from LIGFX import *
from sklearn.metrics import confusion_matrix


def main():
    ligfx_analysis = LIGFX('feature_map.csv')
    ligfx_analysis.holdout()

    ligfx_analysis.create_classifier()
    ligfx_analysis.run_training()
    ligfx_analysis.run_prediction()

    conf_mat = confusion_matrix(ligfx_analysis.test_y, ligfx_analysis.predicted_y)
    print(conf_mat)
    

if __name__ == '__main__':
    main()
