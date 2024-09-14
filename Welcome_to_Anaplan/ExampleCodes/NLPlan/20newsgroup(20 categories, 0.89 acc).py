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
activation_potentiation = ['bent_identity',
                           'waveakt', 
                           'selu', 
                           'gelu', 
                           'srelu',
                           'linear',
                           'tanh',
                           'selu',
                           'gelu',
                           'srelu']

W = plan.fit(x_train, y_train, activation_potentiation=activation_potentiation)

# Modeli test etme
test_model = plan.evaluate(x_test, y_test, show_metrices=True, W=W, activation_potentiation=activation_potentiation)
test_acc_plan = test_model[plan.get_acc()]
print(Fore.GREEN + "\n------PLAN Modeli Sonuçları------" + Fore.RESET)
print(f"PLAN Test Accuracy: {test_acc_plan:.4f}")
print(classification_report(plan.decode_one_hot(y_test), test_model[plan.get_preds()], target_names=newsgroups.target_names))
