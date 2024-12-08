# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 02:36:20 2024

@author: hasan
"""
from colorama import Fore
from sklearn.datasets import load_breast_cancer
from anaplan import plan
import numpy as np
import time

# Breast Cancer veri setini yükleme
data = load_breast_cancer()
X = data.data
y = data.target


# Eğitim, test ve doğrulama verilerini ayırma
x_train, x_test, y_train, y_test = plan.split(X, y, 0.4, 42)

x_train, x_val, y_train, y_val = plan.split(x_train, y_train, 0.2, 42)

# One-hot encoding işlemi
y_train, y_test = plan.encode_one_hot(y_train, y_test)
y_val = plan.encode_one_hot(y_val, y)[0]

x_train, y_train = plan.auto_balancer(x_train, y_train)
x_test, y_test = plan.auto_balancer(x_test, y_test)

scaler_params, x_train, x_test = plan.standard_scaler(x_train, x_test)

#activation_potentiation = plan.activation_optimizer(x_train, y_train, x_test, y_test, early_stop=True)
activation_potentiation = ['tanh']

# Modeli eğitme
W = plan.fit(x_train, y_train, activation_potentiation=activation_potentiation, LTD=2)


# Modeli test etme
test_model = plan.evaluate(x_test, y_test, show_metrics=True, W=W, activation_potentiation=activation_potentiation)

# Modeli kaydetme
plan.save_model(model_name='breast_cancer',
                test_acc=test_model[plan.get_acc()],
                scaler_params=scaler_params,
                activation_potentiation=activation_potentiation,
                W=W)

# Model tahminlerini değerlendirme
for i in range(len(x_val)):
    Predict = plan.predict_model_ssd(model_name='breast_cancer', model_path='', Input=x_val[i])

    time.sleep(0.5)
    if np.argmax(Predict) == np.argmax(y_val[i]):
        print(Fore.GREEN + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', np.argmax(y_val[i]))
    else:
        print(Fore.RED + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', np.argmax(y_val[i]))
