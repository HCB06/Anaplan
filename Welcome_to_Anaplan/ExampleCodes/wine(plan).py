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

model = plan.learner(x_train, y_train, x_test, y_test, show_history=True, depth=2) # learner function = TFL(Test Feedback Learning). If test parameters not given then uses Train Feedback. More information: https://github.com/HCB06/Anaplan/blob/main/Welcome_to_PLAN/PLAN.pdf

W = model[plan.get_weights()]
activation_potentiation = model[plan.get_act_pot()]

test_model = plan.evaluate(x_test, y_test, show_metrics=True, W=W, activation_potentiation=activation_potentiation)

test_preds = test_model[plan.get_preds()]
test_acc = test_model[plan.get_acc()]

model_name = 'wine'
model_path = ''

plan.save_model(model_name=model_name, activation_potentiation=model[plan.get_act_pot()], model_path=model_path, scaler_params=scaler_params, W=model[plan.get_weights()])

precisison, recall, f1 = plan.metrics(y_test, test_preds)
print('Precision: ', precisison, '\n', 'Recall: ', recall, '\n', 'F1: ', f1)
