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

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

newsgroup = fetch_20newsgroups(subset='all', categories=categories)
X = newsgroup.data
y = newsgroup.target

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

X = vectorizer.fit_transform(X)
X = X.toarray()

# Eğitim ve test verilerine ayrıma
x_train, x_test, y_train, y_test = plan.split(X, y, test_size=0.4, random_state=42)


# One-hot encoding işlemi
y_train, y_test = plan.encode_one_hot(y_train, y_test)


# Veri dengesizliği durumunu otomatik dengeleme
x_train, y_train = plan.synthetic_augmentation(x_train, y_train)

scaler_params, x_train, x_test = plan.standard_scaler(x_train, x_test)

print('size of training set: %s' % (len(x_train)))
print('size of validation set: %s' % (len(x_test)))
print('classes: %s' % (newsgroup.target_names))


# PLAN Modeli
model = plan.learner(x_train, y_train, x_test, y_test, depth=4, except_this=['circular'])

W = model[plan.get_weights()]
activation_potentiation = model[plan.get_act_pot()]

test_model = plan.evaluate(x_test, y_test, W=W, activation_potentiation=activation_potentiation)

# Modeli test etme
test_acc_plan = test_model[plan.get_acc()]
print(Fore.GREEN + "\n------PLAN Modeli Sonuçları------" + Fore.RESET)
print(f"PLAN Test Accuracy: {test_acc_plan:.4f}")
print(classification_report(plan.decode_one_hot(y_test), test_model[plan.get_preds()], target_names=newsgroup.target_names))
