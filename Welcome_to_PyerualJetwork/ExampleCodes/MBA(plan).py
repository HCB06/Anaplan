from pyerualjetwork import plan, planeat, data_operations, model_operations, metrics
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

x_train, x_test, y_train, y_test = data_operations.split(X, y, 0.4, 42)
y_train, y_test = data_operations.encode_one_hot(y_train, y_test)

x_train, y_train = data_operations.synthetic_augmentation(x_train, y_train)
x_test, y_test = data_operations.auto_balancer(x_test, y_test)

scaler_params, x_train, x_test= data_operations.standard_scaler(x_train, x_test)

# Configuring optimizator
genetic_optimizer = lambda *args, **kwargs: planeat.evolve(*args, activation_selection_add_prob=0.85, show_info=True, **kwargs)

model = plan.learner(x_train, y_train, genetic_optimizer, x_test, y_test, gen=2, neurons_history=True, auto_normalization=False, except_this=['circular', 'spiral']) # learner function = TFL(Test Feedback Learning). If test parameters not given then uses Train Feedback. More information: https://github.com/HCB06/pyerualjetwork/blob/main/Welcome_to_PLAN/PLAN.pdf

test_model = plan.evaluate(x_test, y_test, W=model[model_operations.get_weights()], show_metrics=True, activation_potentiation=model[model_operations.get_act_pot()])
test_preds = test_model[model_operations.get_preds()]
test_acc = test_model[model_operations.get_acc()]


precisison, recall, f1 = metrics.metrics(y_test, test_preds)
print('Precision: ', precisison, '\n', 'Recall: ', recall, '\n', 'F1: ', f1)
