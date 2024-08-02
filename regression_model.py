from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

data = pd.read_csv('train_regression_model.csv')
train_data = data.iloc[:,:x] # replace x with the specific number of features

x_train, x_test, y_train, y_test = train_test_split(train_data, data.label, test_size = 0.2, random_state = 22)

estimator = RandomForestRegressor()
param_grid = {'max_depth':[5,10,15,20,25,30,35,40,45,50], 'max_features': ['sqrt','log2'], 'min_samples_leaf':[1,2,3,4,5,6,7,8], 'min_samples_split':[2,3,4,5,6,7,8], 'n_estimators':[50,100,150,200,250,300,350,400,450,500]}
estimator = GridSearchCV(estimator,param_grid=param_grid,cv=5)
estimator.fit(x_train,y_train)

best_params = estimator.best_params_
print(best_params)

y_true = y_test
y_predict = estimator.predict(x_test)

MSE = mean_squared_error(y_test, y_predict)
MAE = mean_absolute_error(y_test, y_predict)
R2 = r2_score(y_test, y_predict)