# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 02:36:20 2024

@author: hasan
"""
from colorama import Fore
from sklearn.datasets import make_blobs
from anaplan import plan
import numpy as np
import time

X, y = make_blobs(n_samples=1000, centers=5, random_state=42)


x_train, x_test, y_train, y_test = plan.split(X, y, 0.4, 42)
x_train, x_val, y_train, y_val = plan.split(x_train, y_train, 0.2, 42)

y_train, y_test = plan.encode_one_hot(y_train, y_test)
y_val = plan.encode_one_hot(y_val, y)[0]

x_train, y_train = plan.auto_balancer(x_train, y_train)

scaler_params, x_train, x_test = plan.standard_scaler(x_train, x_test)

activation_potentiation = ['tanh', 'tanh_circular']

W = plan.fit(x_train, y_train, activation_potentiation=activation_potentiation)

test_model = plan.evaluate(x_test, y_test, show_metrices=True, W=W, activation_potentiation=activation_potentiation)

plan.save_model(model_name='blobs',
                model_type='deep PLAN',
                test_acc=test_model[plan.get_acc()],
                weights_type='txt',
                weights_format='f',
                model_path='',
                scaler_params=scaler_params,
                activation_potentiation=activation_potentiation,
                W=W)


for i in range(len(x_val)):
    Predict = plan.predict_model_ssd(model_name='blobs', model_path='', Input=x_val[i])

    time.sleep(0.5)
    if np.argmax(Predict) == np.argmax(y_val[i]):
        print(Fore.GREEN + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', np.argmax(y_val[i]))
    else:
        print(Fore.RED + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', np.argmax(y_val[i]))
