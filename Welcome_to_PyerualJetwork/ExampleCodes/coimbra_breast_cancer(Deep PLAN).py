import pandas as pd
import numpy as np
from colorama import Fore
from pyerualjetwork import plan, planeat, data_operations, model_operations
import time
from sklearn.metrics import classification_report

file_path = 'breast_cancer_coimbra.csv' 
data = pd.read_csv(file_path)


X = data.drop('Classification', axis=1).values
y = data['Classification'].values

# Eğitim, test ve doğrulama verilerini ayırma
x_train, x_test, y_train, y_test = data_operations.split(X, y, 0.4, 42)
x_train, x_val, y_train, y_val = data_operations.split(x_train, y_train, 0.2, 42)

# One-hot encoding işlemi
y_train, y_test = data_operations.encode_one_hot(y_train, y_test)
y_val = data_operations.encode_one_hot(y_val, y)[0]

# Veri dengesizliği durumunda otomatik dengeleme
x_train, y_train = data_operations.auto_balancer(x_train, y_train)

# Verilerin standardize edilmesi
scaler_params, x_train, x_test = data_operations.standard_scaler(x_train, x_test)

# Configuring optimizator
genetic_optimizer = lambda *args, **kwargs: planeat.evolve(*args, **kwargs)

# Training Process
model = plan.learner(x_train, y_train, genetic_optimizer, x_test, y_test, target_acc=0.82)

activation_potentiation = model[model_operations.get_act_pot()]
W = model[model_operations.get_weights()]

# Modeli test etme
test_model = plan.evaluate(x_test, y_test, show_metrics=True,  W=W, activation_potentiation=activation_potentiation)
print(classification_report(plan.decode_one_hot(y_test), test_model[model_operations.get_preds()]))
test_acc = test_model[model_operations.get_acc()]

model_operations.save_model(model_name='breast_cancer_coimbra',
                scaler_params=scaler_params,
                activation_potentiation=activation_potentiation,
                W=W)

# Model tahminlerini değerlendirme
for i in range(len(x_val)):
    Predict = model_operations.predict_model_ssd(model_name='breast_cancer_coimbra', model_path='', Input=x_val[i])

    time.sleep(0.5)
    if np.argmax(Predict) == np.argmax(y_val[i]):
        print(Fore.GREEN + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', np.argmax(y_val[i]))
    else:
        print(Fore.RED + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', np.argmax(y_val[i]))
