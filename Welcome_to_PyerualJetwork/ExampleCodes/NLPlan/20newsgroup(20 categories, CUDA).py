# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 03:55:15 2024

@author: hasan can beydili

'pip install cupy-cuda12x' or your cuda version.

"""

from colorama import Fore
from pyerualjetwork import plan_cuda, data_operations_cuda, model_operations_cuda
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
x_train, x_test, y_train, y_test = data_operations_cuda.split(X, y, test_size=0.2, random_state=42)

# One-hot encoding işlemi
y_train, y_test = data_operations_cuda.encode_one_hot(y_train, y_test)

scaler_params, x_train, x_test = data_operations_cuda.standard_scaler(x_train, x_test)

model = plan_cuda.learner(x_train, y_train, x_test, y_test, depth=2, auto_normalization=False, except_this=['spiral', 'circular']) # learner function = TFL(Test Feedback Learning). If test parameters not given then uses Train Feedback. More information: https://github.com/HCB06/pyerualjetwork/blob/main/Welcome_to_plan_cuda/plan_cuda.pdf

test_acc = model[model_operations_cuda.get_acc()]
test_preds = model[model_operations_cuda.get_preds()]
W = model[model_operations_cuda.get_weights()]
activation_potentiation = model[model_operations_cuda.get_act_pot()]

# Modeli kaydetme
model_operations_cuda.save_model(model_name='20newsgroup', test_acc=test_acc, scaler_params=scaler_params, W=W, activation_potentiation=activation_potentiation)

print(Fore.GREEN + "\n------plan_cuda Modeli Sonuçları------" + Fore.RESET)
print(f"plan_cuda Test Accuracy: {test_acc:.4f}")
print(classification_report(data_operations_cuda.decode_one_hot(y_test), model[model_operations_cuda.get_preds()], target_names=newsgroups.target_names))
