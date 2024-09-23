# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 03:55:15 2024

@author: hasan can beydili
"""

from colorama import Fore
from anaplan import plan
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
import numpy as np

newsgroups = fetch_20newsgroups(subset='all')

X = newsgroups.data  # Metin verisi
y = newsgroups.target  # Sınıf etiketleri

# Metin verilerini TF-IDF özelliğine dönüştürme
vectorizer = TfidfVectorizer(stop_words='english', max_features=18500)

X = vectorizer.fit_transform(X)
X = X.toarray()

# Eğitim ve test verilerine ayırma
x_train, x_test, y_train, y_test = plan.split(X, y, test_size=0.2, random_state=42)

# One-hot encoding işlemi
y_train, y_test = plan.encode_one_hot(y_train, y_test)

# Veri dengesizliği durumunu otomatik dengeleme
x_train, y_train = plan.synthetic_augmentation(x_train, y_train)

scaler_params, x_train, x_test = plan.standard_scaler(x_train, x_test)

print('size of training set: %s' % (len(x_train)))
print('size of validation set: %s' % (len(x_test)))
print('classes: %s' % (newsgroups.target_names))


# PLAN Modeli
model = plan.learner(x_train, y_train, x_test, y_test, depth=10, target_acc=0.89, big_data_mode=True, strategy='accuracy', except_this=['circular'])

activation_potentiation = model[plan.get_act_pot()]
W = model[plan.get_weights()]

W = plan.fit(x_train, y_train, activation_potentiation=activation_potentiation)

# Modeli test etme
test_model = plan.evaluate(x_test, y_test, show_metrics=True, W=W, activation_potentiation=activation_potentiation)
test_acc_plan = test_model[plan.get_acc()]
print(Fore.GREEN + "\n------PLAN Modeli Sonuçları------" + Fore.RESET)
print(f"PLAN Test Accuracy: {test_acc_plan:.4f}")
print(classification_report(plan.decode_one_hot(y_test), test_model[plan.get_preds()], target_names=newsgroups.target_names))
