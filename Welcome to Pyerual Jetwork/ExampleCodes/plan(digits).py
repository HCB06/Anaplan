from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_digits
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import plan_di as pdi
import time
from colorama import Fore


digits = load_digits()
x, y = digits.data, digits.target


encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))


x_train, x_test, y_train, y_test = train_test_split(x, y_onehot, test_size=0.4, random_state=42)


class_count = 10
visualize = 'n'

x_train, y_train = pdi.auto_balancer(x_train, y_train, class_count)
x_test, y_test = pdi.auto_balancer(x_test, y_test, class_count)

W = pdi.fit(x_train, y_train)

test_model = pdi.evaluate(x_test, y_test, visualize, W)
test_preds = test_model[pdi.get_preds()]
test_acc = test_model[pdi.get_acc()]

model_name = 'digits'
model_type = 'PLAN'
weights_type = 'txt'
weights_format = 'd'
model_path = 'PlanModels/'

pdi.save_model(model_name, model_type, class_count, test_acc, weights_type, weights_format, model_path, W)

for i in range(len(x_test)):
    Predict = pdi.predict_model_ram(x_test[i], W)
    image = np.reshape(x_test[i], (8, 8))
    plt.imshow(image)
    plt.title(np.argmax(Predict))
    plt.show()
    time.sleep(0.2)
    if np.argmax(Predict) == np.argmax(y_test[i]):
        print(Fore.GREEN + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', np.argmax(y_test[i]))
    else:
        print(Fore.RED + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', np.argmax(y_test[i]))



y_test_orig = np.argmax(y_test, axis=1)


cm = confusion_matrix(y_test_orig, test_preds)
print("\nConfusion Matrix:")
print(cm)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show()

f1 = f1_score(y_test_orig, test_preds, average='weighted')
print('\nF1 Score:', f1)
