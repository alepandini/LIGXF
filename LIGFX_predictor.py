from LIGFX import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def main():
    ligfx_prediction = LIGFX('feature_map.csv')

    # data=ligfx_prediction.input_data
    # data_norm=ligfx_prediction.norm_data
    #
    # data_vars=data.columns.values.tolist()
    #
    # out='y'
    # columns=[ i for i in data_vars if i not in out]
    #
    #
    # X=data_norm[columns]
    # y=data['y']
    #
    # X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.3,random_state=0)
    # logreg = LogisticRegression()
    # logreg.fit(X_train,y_train)
    #
    # y_pred = logreg.predict(X_test)
    #
    # from sklearn.metrics import confusion_matrix
    #
    #
    # conf_mat = confusion_matrix(y_test,y_pred)
    # print(conf_mat)

if __name__ == '__main__':
    main()
