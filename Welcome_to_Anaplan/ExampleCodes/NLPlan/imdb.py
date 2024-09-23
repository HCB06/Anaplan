from anaplan import plan
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from termcolor import colored
import numpy as np
import time
import pickle

# Veri yükleme ve işleme
data = pd.read_csv('IMDB Dataset.csv') # dataset link: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

X = data['review']
y = data['sentiment']

# Cümlelerin orijinal hallerini kopyalamak için ön ayırma işlemi
x_train, x_test, y_train, y_test = plan.split(X, y, test_size=0.4, random_state=42)

x_test_copy = np.copy(x_test)

# TF-IDF vektörlemesi
vectorizer = TfidfVectorizer(max_features=6000)
X = vectorizer.fit_transform(X)

# Vectorizer'ı kaydetme
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

X = X.toarray()


# Veriyi eğitim ve test olarak ayırma
x_train, x_test, y_train, y_test = plan.split(X, y, test_size=0.4, random_state=42)

# One-hot encoding işlemi
y_train, y_test = plan.encode_one_hot(y_train, y_test)

# Veri dengeleme işlemi
x_train, y_train = plan.synthetic_augmentation(x_train, y_train)
x_test, y_test = plan.auto_balancer(x_test, y_test)

# Veriyi standartlaştırma
scaler_params, x_train, x_test = plan.standard_scaler(x_train, x_test)

model = plan.learner(x_train, y_train, x_test, y_test, target_acc=0.85, big_data_mode=True, except_this=['circular'])

W = model[plan.get_weights()]
activation_potentiation = model[plan.get_act_pot()]

test_model = plan.evaluate(x_test, y_test, W=W, activation_potentiation=activation_potentiation)

# Test sonuçları ve tahminler
test_acc = test_model[plan.get_acc()]
test_preds = test_model[plan.get_preds()]

# Modeli kaydetme
plan.save_model(model_name='IMDB', model_type='PLAN', test_acc=test_acc, weights_type='txt', weights_format='raw', model_path='', scaler_params=scaler_params, W=W, activation_potentiation=activation_potentiation)

# Performans metrikleri
precision, recall, f1 = plan.metrics(y_test, test_preds)

print('Precision: ', precision, '\n', 'Recall: ', recall, '\n', 'F1: ', f1)

y_test = plan.decode_one_hot(y_test)

# Test verisi üzerinde tahminleri yazdırma
for i in range(len(x_test)):
    
    true_label_text = "positive" if y_test[i] == 1 else "negative"
    pred_text = "positive" if test_preds[i] == 1 else "negative"

    time.sleep(1)

    # Tahminin doğru olup olmadığını kontrol etme
    if y_test[i] == test_preds[i]:
        output = colored(f"Review: {x_test_copy[i]}\nPrediction: {pred_text}\nTrue Label: {true_label_text}\n", 'green')
    else:
        output = colored(f"Review: {x_test_copy[i]}\nPrediction: {pred_text}\nTrue Label: {true_label_text}\n", 'red')

    print(output)
