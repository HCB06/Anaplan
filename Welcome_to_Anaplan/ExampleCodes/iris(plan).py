import time
from colorama import Fore
import plan
from sklearn.datasets import load_iris
import numpy as np

data = load_iris()
X = data.data
y = data.target

x_train, x_test, y_train, y_test = plan.split(X, y, 0.3, 42)

x_train = x_train.tolist()
x_test = x_test.tolist()

y_train, y_test = plan.encode_one_hot(y_train, y_test)

x_train, y_train = plan.synthetic_augmentation(x_train, y_train)

scaler_params, x_train, x_test = plan.standard_scaler(x_train, x_test)

model = plan.learner(x_train, y_train, x_test, y_test, neurons_history=True, target_acc=1, interval=16.67) # learner function = TFL(Test Feedback Learning). If test parameters not given then uses Train Feedback. More information: https://github.com/HCB06/Anaplan/blob/main/Welcome_to_PLAN/PLAN.pdf

W = model[plan.get_weights()]

test_model = plan.evaluate(x_test, y_test, W=W, activation_potentiation=model[plan.get_act_pot()])

test_preds = test_model[plan.get_preds()]
test_acc = test_model[plan.get_acc()]

plan.save_model(model_name='iris',
                 model_type='PLAN',
                 test_acc=test_acc,
                 weights_type='txt',
                 weights_format='raw',
                 model_path='',
                 activation_potentiation=model[plan.get_act_pot()],
                 scaler_params=scaler_params,
                 W=W)

precisison, recall, f1 = plan.metrics(y_test, test_preds)
print('Precision: ', precisison, '\n', 'Recall: ', recall, '\n', 'F1: ', f1)

y_test = plan.decode_one_hot(y_test)


for i in range(len(x_test)):
    Predict = plan.predict_model_ram(x_test[i], W=W, activation_potentiation=model[plan.get_act_pot()])
    time.sleep(0.6)
    if np.argmax(Predict) == y_test[i]:
        print(Fore.GREEN + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', y_test[i])
    else:
        print(Fore.RED + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', y_test[i])
