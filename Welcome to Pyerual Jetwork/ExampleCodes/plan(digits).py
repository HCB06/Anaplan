import time
from colorama import Fore, Style
import plan
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt



data = load_digits()
X = data.data
y = data.target

X = plan.normalization(X)
    
x_train, x_test, y_train, y_test = plan.split(X, y, 0.4, 42)
    
    
x_train = x_train.tolist()
x_test = x_test.tolist()


y_train, y_test = plan.encode_one_hot(y_train, y_test)

show_metrics = True
scaler_params = None
class_count = 10
show_training = None
show_count = None
val = True

x_test, y_test = plan.auto_balancer(x_test, y_test)


W = plan.fit(x_train, y_train, show_training, show_count, val)

test_model = plan.evaluate(x_test, y_test, show_metrics, W)
test_preds = test_model[plan.get_preds()]
test_acc = test_model[plan.get_acc()]

model_name = 'digits'
model_type = 'PLAN'
weights_type = 'txt'
weights_format = 'd'
model_path = 'PlanModels'

plan.save_model(model_name, model_type, class_count, test_acc, weights_type, weights_format, model_path, scaler_params, W)

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
