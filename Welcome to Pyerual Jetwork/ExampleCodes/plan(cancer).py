from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.datasets import load_breast_cancer
import plan as p
import numpy as np
from sklearn.metrics import f1_score
import time
from colorama import Fore

data = load_breast_cancer()
X = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


encoder = OneHotEncoder()
y_train = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
y_test = encoder.transform(y_test.reshape(-1, 1)).toarray()


x_train = x_train.tolist()
x_test = x_test.tolist()

activation_potential = 0.1
visualize = 'n'

train_model = p.fit(x_train, y_train, activation_potential)
W = train_model[p.get_weights()]


test_model = p.evaluate(x_test, y_test, activation_potential, visualize, W)
test_preds = test_model[p.get_preds()]


for i in range(len(x_test)):
    Predict = p.predict_model_ram(x_test[i], activation_potential, W)
    time.sleep(0.6)
    if np.argmax(Predict) == y_test[i]:
        print(Fore.GREEN + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', y_test[i])
    else:
        print(Fore.RED + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', y_test[i])
    


y_test = np.argmax(y_test, axis=1)

f1 = f1_score(y_test, test_preds, average='weighted')
print('\nF1 Score:', f1)