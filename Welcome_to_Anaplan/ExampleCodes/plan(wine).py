# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 23:32:16 2024

@author: hasan
"""

import time
from colorama import Fore
from anaplan import plan
from sklearn.datasets import load_wine
import numpy as np

data = load_wine()
X = data.data
y = data.target

x_train, x_test, y_train, y_test = plan.split(X, y, 0.4, 42)

y_train, y_test = plan.encode_one_hot(y_train, y_test)

x_test, y_test = plan.auto_balancer(x_test, y_test)

scaler_params, x_train, x_test = plan.standard_scaler(x_train, x_test)

act_pot = plan.activation_optimizer(x_train, y_train, x_test, y_test, early_stop=True)

W = plan.fit(x_train, y_train, activation_potentiation=act_pot)

test_model = plan.evaluate(x_test, y_test, show_metrics=True, W=W, activation_potentiation=act_pot)
test_preds = test_model[plan.get_preds()]
test_acc = test_model[plan.get_acc()]

model_name = 'wine'
model_type = 'PLAN'
weights_type = 'txt'
weights_format = 'raw'
model_path = ''

plan.save_model(model_name=model_name, model_type=model_type, test_acc=test_acc, weights_type=weights_type, weights_format=weights_format, model_path=model_path, scaler_params=scaler_params, W=W)

precisison, recall, f1 = plan.metrics(y_test, test_preds)
print('Precision: ', precisison, '\n', 'Recall: ', recall, '\n', 'F1: ', f1)
