import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


data=pd.read_csv('feature_map_con_rms.csv',header=0)
data_norm=data.apply(lambda x: (x - np.mean(x)) / ( np.max(x) - np.min(x)))

data_vars=data.columns.values.tolist()

out='y'
columns=[ i for i in data_vars if i not in out]


X=data_norm[columns]
y=data['y']

X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.3,random_state=20)
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train,y_train)

y_pred = svclassifier.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix,r2_score,f1_score,accuracy_score

'''
conf_mat = confusion_matrix(y_test,y_pred)
print(conf_mat)
print(classification_report(y_test,y_pred))
'''


for i in range(10):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
	svclassifier = SVC(kernel='linear')
	svclassifier.fit(X_train,y_train)
	
	y_pred = svclassifier.predict(X_test)
	print (f1_score(y_test,y_pred))
