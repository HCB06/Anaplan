import pandas as pd
import numpy as np
from colorama import Fore
import plan
import time
from sklearn.metrics import classification_report

file_path = 'breast_cancer_coimbra.csv' 
data = pd.read_csv(file_path)


X = data.drop('Classification', axis=1).values
y = data['Classification'].values

# Eğitim, test ve doğrulama verilerini ayırma
x_train, x_test, y_train, y_test = plan.split(X, y, 0.4, 42)
x_train, x_val, y_train, y_val = plan.split(x_train, y_train, 0.2, 42)

# One-hot encoding işlemi
y_train, y_test = plan.encode_one_hot(y_train, y_test)
y_val = plan.encode_one_hot(y_val, y)[0]

# Veri dengesizliği durumunda otomatik dengeleme
x_train, y_train = plan.auto_balancer(x_train, y_train)

# Verilerin standardize edilmesi
scaler_params, x_train, x_test = plan.standard_scaler(x_train, x_test)

# Aktivasyon fonksiyonları
activation_potentiation = [None, 'waveakt']

# Modeli eğitme
W = plan.fit(x_train, y_train, activation_potentiation=activation_potentiation, LTD=0) # val=True, show_training=True, val_count=(int), interval=(int), x_val=(default: x_train), y_val=(default: y_train)

# Modeli test etme
test_model = plan.evaluate(x_test, y_test, show_metrices=True,  W=W, activation_potentiation=activation_potentiation)
print(classification_report(plan.decode_one_hot(y_test), test_model[plan.get_preds()]))
test_acc = test_model[plan.get_acc()]

plan.save_model(model_name='breast_cancer_coimbra',
                model_type='deep PLAN',
                test_acc=test_acc,
                weights_type='txt',
                weights_format='f',
                model_path='',
                scaler_params=scaler_params,
                activation_potentiation=activation_potentiation,
                W=W)

# Model tahminlerini değerlendirme
for i in range(len(x_val)):
    Predict = plan.predict_model_ssd(model_name='breast_cancer_coimbra', model_path='', Input=x_val[i])

    time.sleep(0.5)
    if np.argmax(Predict) == np.argmax(y_val[i]):
        print(Fore.GREEN + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', np.argmax(y_val[i]))
    else:
        print(Fore.RED + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', np.argmax(y_val[i]))
