import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


data=pd.read_csv('feature_map_con_rms.csv',header=0)
data_norm=data.apply(lambda x: (x - np.mean(x)) / ( np.max(x) - np.min(x)))

data_vars=data.columns.values.tolist()

out='y'
columns=[ i for i in data_vars if i not in out]


X=data_norm[columns]
y=data['y']

X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.3,random_state=20)
logreg = LogisticRegression()
logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix,r2_score,f1_score,accuracy_score


#conf_mat = confusion_matrix(y_test,y_pred)
#print(conf_mat)
#print(classification_report(y_test,y_pred))
#print(r2_score(y_test,y_pred))

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)
    print (accuracy_score(y_test,y_pred))
'''

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# feature extraction
data_pos = data.apply(lambda x: np.abs(x))
X=data_pos[columns]
test = SelectKBest(score_func=chi2, k=4)
fitted = test.fit(X, y)

np.set_printoptions(precision=3)
coppia= []
for i in range(len(fitted.scores_)):
    coppia.append((columns[i],fitted.scores_[i]))

coppia_sort = sorted(coppia, key=lambda x: x[1])

for ele in coppia_sort:
    print (ele)
'''
