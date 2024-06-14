import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import plan_bi as pbi
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from colorama import Fore
import seaborn as sns
import matplotlib.pyplot as plt
import time


data = pd.read_csv('creditcard.csv')

x = data.drop('Class', axis=1).values
y = data['Class'].values


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

encoder = OneHotEncoder(sparse_output=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.transform(y_test.reshape(-1, 1))


x_train = x_train.tolist()
x_test = x_test.tolist()

x_train, x_test = pbi.standard_scaler(x_train, x_test)

activation_potential = 0
class_count = 2

x_train, y_train = pbi.auto_balancer(x_train, y_train, class_count)
x_test, y_test = pbi.synthetic_augmentation(x_test, y_test, class_count)

W = pbi.fit(x_train, y_train, activation_potential)

visualize = 'n'

test_model = pbi.evaluate(x_test, y_test, activation_potential, visualize, W)

test_preds = test_model[pbi.get_preds()]
test_acc = test_model[pbi.get_acc()]

model_name = 'creditcard_fraud'
model_type = 'PLAN'
weights_type = 'txt'
weights_format = 'd'
model_path = 'PlanModels/'

pbi.save_model(model_name, model_type, class_count, activation_potential, test_acc, weights_type, weights_format, model_path, W)


y_test = np.argmax(y_test, axis=1)

"""

for i in range(len(x_test)):
    Predict = pbi.predict_model_ssd(x_test[i], model_name, model_path)
    time.sleep(0.6)
    if np.argmax(Predict) == y_test[i]:
        print(Fore.GREEN + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', y_test[i])
    else:
        print(Fore.RED + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', y_test[i])

"""

cm = confusion_matrix(y_test, test_preds)
print("\nConfusion Matrix:")
print(cm)


sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show()


"""
# Commented out prediction loop...
"""

f1 = f1_score(y_test, test_preds, average='weighted')
print('\nF1 Score:', f1)
