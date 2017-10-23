from xgboost.sklearn import XGBRegressor
from sklearn.metrics import r2_score
import pandas as pd


data = pd.read_csv('train.csv').drop(['Id'], axis=1)

data_corr = data.corr().abs()
important_columns = [col for col in data_corr.loc[:, data_corr['SalePrice'] >= 0.5].columns]

train_data = data[:int(data.shape[0]*0.7)][important_columns]
test_data = data[train_data.shape[0]:][important_columns]

model = XGBRegressor()

x_train = train_data.drop(['SalePrice'], axis=1)
y_train = train_data['SalePrice']
model.fit(x_train, y_train)

x_test = test_data.drop(['SalePrice'], axis=1)
y_test = test_data['SalePrice']

predictions = model.predict(x_test)
print('Score: {}%'.format(int(r2_score(y_test, predictions)*100)))
