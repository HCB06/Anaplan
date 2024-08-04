import plan
import time
from colorama import Fore
import numpy as np
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt

# TRAIN

data = load_digits()

X = data.data
y = data.target

X = plan.normalization(X)
    
x_train, x_test, y_train, y_test = plan.split(X, y, 0.4, 42)


y_train, y_test = plan.encode_one_hot(y_train, y_test)


x_test, y_test = plan.auto_balancer(x_test, y_test)

plan.plot_decision_space(x_train, plan.decode_one_hot(y_train))
plt.show()

W = plan.fit(x_train, y_train)

# TEST

test_model = plan.evaluate(x_train, y_train, show_metrices=True, W=W)
test_preds = test_model[plan.get_preds()]
test_acc = test_model[plan.get_acc()]

# PREDICT

for i in range(len(x_test)):
    Predict = plan.predict_model_ram(x_test[i], W)

    time.sleep(0.5)
    if np.argmax(Predict) == np.argmax(y_test[i]):
        print(Fore.GREEN + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', np.argmax(y_test[i]))
    else:
        print(Fore.RED + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', np.argmax(y_test[i]))
