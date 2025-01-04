import pandas as pd
from anaplan import plan, data_operations

data = pd.read_csv('creditcard.csv') # dataset link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

x = data.drop('Class', axis=1).values
y = data['Class'].values

x = plan.normalization(x)
x_train, x_test, y_train, y_test = data_operations.split(x, y, 0.4, 42)

y_train, y_test = data_operations.encode_one_hot(y_train, y_test)


scaler_params, x_train, x_test = data_operations.standard_scaler(x_train, x_test)

x_train, y_train = data_operations.synthetic_augmentation(x_train, y_train)

model = plan.learner(x_train, y_train, x_test, y_test, batch_size=0.1, auto_normalization=False, target_acc=0.99) # learner function = TFL(Test Feedback Learning). If test parameters not given then uses Train Feedback. More information: https://github.com/HCB06/anaplan/blob/main/Welcome_to_PLAN/PLAN.pdf

activation_potentiation = model[plan.get_act_pot()]
W = model[plan.get_weights()]

test_model = plan.evaluate(x_test, y_test, show_metrics=True, W=W, activation_potentiation=activation_potentiation)
