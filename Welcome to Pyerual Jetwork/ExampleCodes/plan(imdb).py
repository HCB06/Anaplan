import plan_di as pdi
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


data = pd.read_csv('IMDB Dataset.csv')


X = data['review']
y = data['sentiment']

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(X)
X = X.toarray()

x_train, x_test, y_train, y_test = pdi.split(X, y, 0.2, 42)

x_train = np.array(x_train)
x_test = np.array(x_test)

y_train = np.array(y_train)
y_test = np.array(y_test)

y_train, y_test = pdi.encode_one_hot(y_train, y_test)

x_train.tolist()
x_test.tolist()

show_metrices = True

x_train, y_train = pdi.auto_balancer(x_train, y_train)
x_test, y_test = pdi.auto_balancer(x_test, y_test)
x_train, x_test = pdi.standard_scaler(x_train, x_test)
W = pdi.fit(x_train, y_train)

test_model = pdi.evaluate(x_test, y_test, show_metrices, W)

test_acc = test_model[pdi.get_acc()]
test_preds = test_model[pdi.get_preds()]

model_name = 'IMDB'
model_type = 'PLAN'
class_count = 2
weights_type = 'txt'
weights_format = 'd'
model_path = 'PlanModels/'
scaler = True

pdi.save_model(model_name, model_type, class_count, test_acc, weights_type, weights_format, model_path, scaler, W)

precisison, recall, f1 = pdi.metrics(y_test, test_preds)

print('Precision: ', precisison, '\n', 'Recall: ', recall, '\n', 'F1: ', f1)