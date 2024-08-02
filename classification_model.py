from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, matthews_corrcoef, confusion_matrix
import pandas as pd
from joblib import dump

data = pd.read_csv('train_classification_model.csv')
train_data = data.iloc[:,:x] # replace x with the specific number of features

x_train, x_test, y_train, y_test = train_test_split(train_data, data.label, test_size = 0.2, random_state = 22)

estimator = RandomForestClassifier()
param_grid = {'max_depth':[5,10,15,20,25,30,35,40,45,50], 'min_samples_leaf':[1,2,3,4,5,6,7,8], 'min_samples_split':[2,3,4,5,6,7,8], 'n_estimators':[50,100,150,200,250,300,350,400,450,500]}
estimator = GridSearchCV(estimator,param_grid=param_grid,cv=5)
estimator.fit(x_train,y_train)

best_params = estimator.best_params_
print(best_params)

y_true = y_test
y_predict = estimator.predict(x_test)

accuracy = accuracy_score(y_true, y_predict)
print(accuracy)
sensitivity = recall_score(y_true, y_predict)
print(sensitivity)
cm = confusion_matrix(y_true, y_predict)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
print(specificity)
MCC = matthews_corrcoef(y_true, y_predict)
print(MCC)