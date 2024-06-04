import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import f1_score
import plan
import time
from colorama import Fore

wine = load_wine()
X = wine.data
y = wine.target

one_hot_encoder = OneHotEncoder(sparse=False)
y = one_hot_encoder.fit_transform(y.reshape(-1, 1))


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.tolist()
x_test = x_test.tolist()
y_train = y_train.tolist()
y_test = y_test.tolist()

activation_potential = 0
class_count = 3
visualize = 'n'

#x_train, y_train = plan.synthetic_augmentation(x_train, y_train, class_count)
#x_test, y_test = plan.synthetic_augmentation(x_test, y_test, class_count)

train_model = plan.fit(x_train, y_train, activation_potential)
W = train_model[plan.get_weights()]

test_model = plan.evaluate(x_test, y_test, activation_potential, visualize, W)
test_preds = test_model[plan.get_preds()]
test_acc = test_model[plan.get_acc()]

y_test = np.array(y_test)


y_test = np.argmax(y_test, axis=1)

f1 = f1_score(y_test, test_preds, average='weighted')
print('\nF1 Score:', f1)



for i in range(len(x_test)):
    Predict = plan.predict_model_ram(x_test[i], activation_potential, W)
    time.sleep(0.6)
    if np.argmax(Predict) == y_test[i]:
        print(Fore.GREEN + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', y_test[i])
    else:
        print(Fore.RED + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', y_test[i])