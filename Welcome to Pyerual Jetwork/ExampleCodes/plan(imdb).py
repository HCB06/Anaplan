import plan
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

data = pd.read_csv('IMDB Dataset.csv')

X = data['review']
y = data['sentiment']

vectorizer = TfidfVectorizer(max_features=6084)
X = vectorizer.fit_transform(X)
X = X.toarray()

x_train, x_test, y_train, y_test = plan.split(X, y, 0.2, 42)

x_train = x_train.tolist()
x_test = x_test.tolist()

y_train = np.array(y_train)
y_test = np.array(y_test)

y_train, y_test = plan.encode_one_hot(y_train, y_test)

scaler_params, x_train, x_test = plan.standard_scaler(x_train, x_test)

show_metrices = True
show_training = 'final' # other values: True or None(any)

x_train, y_train = plan.auto_balancer(x_train, y_train)

W = plan.fit(x_train, y_train, show_training)


test_model = plan.evaluate(x_test, y_test, show_metrices, W)


test_acc = test_model[plan.get_acc()]
test_preds = test_model[plan.get_preds()]

model_name = 'IMDB'
model_type = 'PLAN'
class_count = 2
weights_type = 'txt'
weights_format = 'd'
model_path = 'PlanModels'
scaler = True

plan.save_model(model_name, model_type, class_count, test_acc, weights_type, weights_format, model_path, scaler, W)

precisison, recall, f1 = plan.metrics(y_test, test_preds)

print('Precision: ', precisison, '\n', 'Recall: ', recall, '\n', 'F1: ', f1)
