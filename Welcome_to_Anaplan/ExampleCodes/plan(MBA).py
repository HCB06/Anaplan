from anaplan import plan
import numpy as np
import pandas as pd

df = pd.read_csv('MBA.csv') # dataset link: https://www.kaggle.com/datasets/taweilo/mba-admission-dataset/data

y = df['international']

X = df.drop(columns=['international'], axis=1)


# Kategorik sütunları seçme
categorical_columns = X.select_dtypes(include=['object']).columns

# One-Hot Encoding
X = pd.get_dummies(X, columns=categorical_columns)


# Bilinmeyen değerleri "?" olan yerleri NaN ile değiştirme
X.replace('?', np.nan, inplace=True)


X.dropna(inplace=True)

X = np.array(X)

x_train, x_test, y_train, y_test = plan.split(X, y, 0.4, 42)
y_train, y_test = plan.encode_one_hot(y_train, y_test)

x_train, y_train = plan.synthetic_augmentation(x_train, y_train)
x_test, y_test = plan.auto_balancer(x_test, y_test)

scaler_params, x_train, x_test= plan.standard_scaler(x_train, x_test)

model = plan.learner(x_train, y_train, x_test, y_test, target_acc=1, big_data_mode=True, except_this=['circular']) # learner function = TFL(Test Feedback Learning). If test parameters not given then uses Train Feedback. More information: https://github.com/HCB06/Anaplan/blob/main/Welcome_to_PLAN/PLAN.pdf

test_model = plan.evaluate(x_test, y_test, W=model[plan.get_weights()], activation_potentiation=model[plan.get_act_pot()])
test_preds = test_model[plan.get_preds()]
test_acc = test_model[plan.get_acc()]


precisison, recall, f1 = plan.metrics(y_test, test_preds)
print('Precision: ', precisison, '\n', 'Recall: ', recall, '\n', 'F1: ', f1)
