import plan
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv('IMDB Dataset.csv') # dataset link: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

X = data['review']
y = data['sentiment']

vectorizer = TfidfVectorizer(max_features=6084)
X = vectorizer.fit_transform(X)
X = X.toarray()

x_train, x_test, y_train, y_test = plan.split(X, y, 0.2, 42)

y_train, y_test = plan.encode_one_hot(y_train, y_test)

scaler_params, x_train, x_test = plan.standard_scaler(x_train, x_test)

x_train, y_train = plan.auto_balancer(x_train, y_train)

W = plan.fit(x_train, y_train)

test_model = plan.evaluate(x_test, y_test, show_metrices=True, W=W)


test_acc = test_model[plan.get_acc()]
test_preds = test_model[plan.get_preds()]

plan.save_model(model_name='IMDB', model_type='PLAN', test_acc=test_acc, weights_type='txt', weights_format='f', model_path='', scaler_params=scaler_params, W=W)

precisison, recall, f1 = plan.metrics(y_test, test_preds)

print('Precision: ', precisison, '\n', 'Recall: ', recall, '\n', 'F1: ', f1)
