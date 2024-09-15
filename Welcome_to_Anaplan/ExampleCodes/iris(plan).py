import time
from colorama import Fore
from anaplan import plan
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

activation_potentiation = plan.activation_optimizer(x_train, y_train, x_test, y_test, target_acc=1)

W = plan.fit(x_train, y_train, activation_potentiation=activation_potentiation)

test_model = plan.evaluate(x_test, y_test, show_metrics=True, W=W, activation_potentiation=activation_potentiation)
test_preds = test_model[plan.get_preds()]
test_acc = test_model[plan.get_acc()]

model_name = 'iris'
model_type = 'PLAN'
weights_type = 'txt'
weights_format = 'raw'
model_path = ''

plan.save_model(model_name, model_type, test_acc, weights_type, weights_format, model_path, scaler_params, W)

precisison, recall, f1 = plan.metrics(y_test, test_preds)
print('Precision: ', precisison, '\n', 'Recall: ', recall, '\n', 'F1: ', f1)

y_test = plan.decode_one_hot(y_test)

"""
for i in range(len(x_test)):
    Predict = plan.predict_model_ram(x_test[i], scaler, W)
    time.sleep(0.6)
    if np.argmax(Predict) == y_test[i]:
        print(Fore.GREEN + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', y_test[i])
    else:
        print(Fore.RED + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', y_test[i])
        
        """
