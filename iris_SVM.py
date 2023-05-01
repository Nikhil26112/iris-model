import pandas as pd
import sklearn
from sklearn.svm import SVC
import pickle
import joblib
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('iris.data', header=None)

X = dataset.iloc[:, 0:4]
y = dataset.iloc[:, 4]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
classifier = SVC(kernel = 'linear', random_state = 0)
#Fit the model on training data
classifier.fit(X_train, y_train)
#Make the prediction
y_pred = classifier.predict(X_test)

filename= 'iris.pkl'
joblib.dump(classifier,open(filename, 'wb'))