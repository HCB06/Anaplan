import pandas as pd
import plan_di as pdi
import numpy as np

data = pd.read_csv('creditcard.csv')

x = data.drop('Class', axis=1).values
y = data['Class'].values


x_train, x_test, y_train, y_test = pdi.split(x, y, 0.4, 42)

y_train, y_test = pdi.encode_one_hot(y_train, y_test)

x_train = x_train.tolist()
x_test = x_test.tolist()

class_count = 2

x_train, x_test = pdi.standard_scaler(x_train, x_test)

x_train, y_train = pdi.auto_balancer(x_train, y_train)
x_test, y_test = pdi.synthetic_augmentation(x_test, y_test)



W = pdi.fit(x_train, y_train)

show_metrics = True

test_model = pdi.evaluate(x_test, y_test, show_metrics, W)

test_preds = test_model[pdi.get_preds()]
test_acc = test_model[pdi.get_acc()]

model_name = 'creditcard_fraud(di)'
model_type = 'PLAN'
weights_type = 'txt'
weights_format = 'd'
model_path = 'PlanModels/'
scaler = True

pdi.save_model(model_name, model_type, class_count, test_acc, weights_type, weights_format, model_path, scaler, W)




"""
y_test = np.argmax(y_test, axis=1)
for i in range(len(x_test)):
    Predict = plan.predict_model_ssd(x_test[i], model_name, model_path)
    time.sleep(0.6)
    if np.argmax(Predict) == y_test[i]:
        print(Fore.GREEN + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', y_test[i])
    else:
        print(Fore.RED + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', y_test[i])

"""

precisison, recall, f1 = pdi.metrics(y_test, test_preds)

print('Precision: ', precisison, '\n', 'Recall: ', recall, '\n', 'F1: ', f1)

