# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 23:32:16 2024

@author: hasan
"""

import time
from colorama import Fore
from pyerualjetwork import plan, planeat, data_operations, model_operations, metrics
from sklearn.datasets import load_wine
import numpy as np

data = load_wine()
X = data.data
y = data.target

x_train, x_test, y_train, y_test = data_operations.split(X, y, 0.4, 42)

y_train, y_test = data_operations.encode_one_hot(y_train, y_test)

x_test, y_test = data_operations.auto_balancer(x_test, y_test)

scaler_params, x_train, x_test = data_operations.standard_scaler(x_train, x_test)

# Configuring optimizer
genetic_optimizer = lambda *args, **kwargs: planeat.evolver(*args, **kwargs)
model = plan.learner(x_train, y_train, genetic_optimizer, x_test, y_test, show_history=True, depth=2) # learner function = TFL(Test Feedback Learning). If test parameters not given then uses Train Feedback. More information: https://github.com/HCB06/pyerualjetwork/blob/main/Welcome_to_PLAN/PLAN.pdf

W = model[model_operations.get_weights()]
activation_potentiation = model[model_operations.get_act_pot()]

test_model = plan.evaluate(x_test, y_test, show_metrics=True, W=W, activation_potentiation=activation_potentiation)

test_preds = test_model[model_operations.get_preds()]
test_acc = test_model[model_operations.get_acc()]

model_name = 'wine'
model_path = ''

model_operations.save_model(model_name=model_name, activation_potentiation=model[model_operations.get_act_pot()], model_path=model_path, scaler_params=scaler_params, W=model[model_operations.get_weights()])

precisison, recall, f1 = metrics(y_test, test_preds)
print('Precision: ', precisison, '\n', 'Recall: ', recall, '\n', 'F1: ', f1)
