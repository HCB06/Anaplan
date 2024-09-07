# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 03:55:15 2024

@author: hasan
"""

import pandas as pd
import numpy as np
from colorama import Fore
from anaplan import plan
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, accuracy_score

# Heart Disease veri setini yükle
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 
           'slope', 'ca', 'thal', 'target']

data = pd.read_csv(url, header=None, names=columns, na_values="?")

# Eksik verileri doldur (bu örnekte ortalama ile)
data.fillna(data.mean(), inplace=True)

# Hedef değişkeni 1 ve 0 olarak ayarla (1: kalp hastalığı var, 0: yok)
data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)

# Özellikleri ve hedef değişkeni ayır
X = data.drop('target', axis=1).values
y = data['target'].values

# Eğitim, test ve doğrulama verilerini ayırma
x_train, x_test, y_train, y_test = plan.split(X, y, 0.2, 42) # For less data use this: (X, y, 0.9, 42)

# One-hot encoding işlemi
y_train, y_test = plan.encode_one_hot(y_train, y_test)

# Veri dengesizliğini düzeltme ve veriyi standardize etme
x_train, y_train = plan.auto_balancer(x_train, y_train)

scaler_params, x_train, x_test = plan.standard_scaler(x_train, x_test)

# Lojistik Regresyon Modeli
print(Fore.YELLOW + "------Lojistik Regresyon Sonuçları------" + Fore.RESET)
lr_model = LogisticRegression(max_iter=1000, random_state=42)
y_train_decoded = plan.decode_one_hot(y_train)
lr_model.fit(x_train, y_train_decoded)

y_test_decoded = plan.decode_one_hot(y_test)
y_pred_lr = lr_model.predict(x_test)
test_acc_lr = accuracy_score(y_test_decoded, y_pred_lr)
print(f"Lojistik Regresyon Test Accuracy: {test_acc_lr:.4f}")
print(classification_report(y_test_decoded, y_pred_lr))

# Random Forest Modeli
print(Fore.CYAN + "------Random Forest Sonuçları------" + Fore.RESET)
rf_model = RandomForestClassifier(n_estimators=1000, random_state=42)
rf_model.fit(x_train, y_train_decoded)

y_pred_rf = rf_model.predict(x_test)
test_acc_rf = accuracy_score(y_test_decoded, y_pred_rf)
print(f"Random Forest Test Accuracy: {test_acc_rf:.4f}")
print(classification_report(y_test_decoded, y_pred_rf))

# XGBoost Modeli
print(Fore.MAGENTA + "------XGBoost Sonuçları------" + Fore.RESET)
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(x_train, y_train_decoded)

y_pred_xgb = xgb_model.predict(x_test)
test_acc_xgb = accuracy_score(y_test_decoded, y_pred_xgb)
print(f"XGBoost Test Accuracy: {test_acc_xgb:.4f}")
print(classification_report(y_test_decoded, y_pred_xgb))

# Derin Öğrenme Modeli (Yapay Sinir Ağı)

input_dim = x_train.shape[1]  # Giriş boyutu

model = Sequential()
model.add(Dense(64, input_dim=input_dim, activation='sigmoid'))  # Giriş katmanı ve ilk gizli katman
model.add(Dropout(0.5))  # Overfitting'i önlemek için Dropout katmanı
model.add(Dense(128, activation='sigmoid'))  # İkinci gizli katman
model.add(Dropout(0.5))  # Overfitting'i önlemek için Dropout katmanı
model.add(Dense(64, activation='sigmoid'))  # Üçüncü gizli katman
model.add(Dropout(0.5))  # Overfitting'i önlemek için Dropout katmanı
model.add(Dense(128, activation='sigmoid'))  # Dördüncü gizli katman
model.add(Dropout(0.5))  # Overfitting'i önlemek için Dropout katmanı
model.add(Dense(y_train.shape[1], activation='softmax'))  # Çıkış katmanı (softmax)

# Modeli derleme
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
history = model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=2)

# Test verileri üzerinde modelin performansını değerlendirme
y_pred_dl = model.predict(x_test)
y_pred_dl_classes = np.argmax(y_pred_dl, axis=1)  # Tahmin edilen sınıflar
y_test_decoded_dl = plan.decode_one_hot(y_test)

print(Fore.BLUE + "------Derin Öğrenme (ANN) Sonuçları------" + Fore.RESET)
test_acc_dl = accuracy_score(y_test_decoded_dl, y_pred_dl_classes)
print(f"Derin Öğrenme Test Accuracy: {test_acc_dl:.4f}")
print(classification_report(y_test_decoded_dl, y_pred_dl_classes))

# PLAN Modeli
print(Fore.GREEN + "------PLAN Modeli Sonuçları------" + Fore.RESET)
activation_potentiation = ['sine', 'waveakt']
W = plan.fit(x_train, y_train, activation_potentiation=activation_potentiation)

# Modeli test etme
test_model = plan.evaluate(x_test, y_test, W=W, activation_potentiation=activation_potentiation)
test_acc_plan = test_model[plan.get_acc()]
print(f"PLAN Test Accuracy: {test_acc_plan:.4f}")
print(classification_report(plan.decode_one_hot(y_test), test_model[plan.get_preds()]))

"""
# MODEL KAYDETME
plan.save_model(model_name='heart_disease_model',
                model_type='deep PLAN',
                test_acc=test_acc_plan,
                weights_type='txt',
                weights_format='raw',
                model_path='',
                scaler_params=scaler_params,
                activation_potentiation=activation_potentiation,
                W=W)
"""
