# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 03:55:15 2024

@author: hasan can beydili
"""

from colorama import Fore
from anaplan import plan, data_operations, model_operations
from sklearn.metrics import classification_report
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

# Vectorizer'ı kaydetme
with open('tfidf_20news.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Eğitim ve test verilerine ayırma
x_train, x_test, y_train, y_test = data_operations.split(X, y, test_size=0.2, random_state=42)

# One-hot encoding işlemi
y_train, y_test = data_operations.encode_one_hot(y_train, y_test)

# Veri dengesizliği durumunu otomatik dengeleme
x_train, y_train = data_operations.synthetic_augmentation(x_train, y_train)

scaler_params, x_train, x_test = data_operations.standard_scaler(x_train, x_test)

print('size of training set: %s' % (len(x_train)))
print('size of validation set: %s' % (len(x_test)))
print('classes: %s' % (newsgroups.target_names))


# PLAN Modeli
model = plan.learner(x_train, y_train, x_test, y_test, depth=10, target_acc=0.89, big_data_mode=True) # learner function = TFL(Test Feedback Learning). If test parameters not given then uses Train Feedback. More information: https://github.com/HCB06/anaplan/blob/main/Welcome_to_PLAN/PLAN.pdf

test_acc = model[model_operations.get_acc()]
test_preds = model[model_operations.get_preds()]
W = model[model_operations.get_weights()]
activation_potentiation = model[model_operations.get_act_pot()]

# Modeli kaydetme
model_operations.save_model(model_name='20newsgroup', test_acc=test_acc, scaler_params=scaler_params, W=W, activation_potentiation=activation_potentiation)

print(Fore.GREEN + "\n------PLAN Modeli Sonuçları------" + Fore.RESET)
print(f"PLAN Test Accuracy: {test_acc:.4f}")
print(classification_report(data_operations.decode_one_hot(y_test), model[model_operations.get_preds()], target_names=newsgroups.target_names))
