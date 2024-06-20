import time
from colorama import Fore
import plan
from sklearn.datasets import load_breast_cancer
import numpy as np

data = load_breast_cancer()
X = data.data
y = data.target

x_train, x_test, y_train, y_test = plan.split(X, y, 0.4, 42)

x_train = x_train.tolist()
x_test = x_test.tolist()


scaler_params, x_train, x_test = plan.standard_scaler(x_train, x_test)

y_train, y_test = plan.encode_one_hot(y_train, y_test)

show_metrics = True
show_training = None
class_count = 2
x_test, y_test = plan.auto_balancer(x_test, y_test)

W = plan.fit(x_train, y_train, show_training)
W = plan.weight_normalization(W, class_count)
test_model = plan.evaluate(x_test, y_test, show_metrics, W)
test_preds = test_model[plan.get_preds()]
test_acc = test_model[plan.get_acc()]

model_name = 'breast_cancer'
model_type = 'PLAN'
weights_type = 'txt'
weights_format = 'f'
model_path = 'PlanModels/'
class_count = 2


scaler_params = None # because x_test is already scaled. If you change this row model make wrong predicts for predect from ssd function. if you just want save model then delete this row. If you want prediction from x_test, don't change.

plan.save_model(model_name, model_type, class_count, test_acc, weights_type, weights_format, model_path, scaler_params, W)


precisison, recall, f1 = plan.metrics(y_test, test_preds)
print('Precision: ', precisison, '\n', 'Recall: ', recall, '\n', 'F1: ', f1)

y_test = plan.decode_one_hot(y_test)


for i in range(len(x_test)):
    Predict = plan.predict_model_ssd(x_test[i], model_name, model_path)
    time.sleep(0.6)
    if np.argmax(Predict) == y_test[i]:
        print(Fore.GREEN + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', y_test[i])
    else:
        print(Fore.RED + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', y_test[i])
        
        
