import pandas as pd
import plan_bi as pbi
import numpy as np

data = pd.read_csv('creditcard.csv')

x = data.drop('Class', axis=1).values
y = data['Class'].values


x_train, x_test, y_train, y_test = pbi.split(x, y, 0.4, 42)

y_train, y_test = pbi.encode_one_hot(y_train, y_test)

x_train = x_train.tolist()
x_test = x_test.tolist()

activation_potential = 0
class_count = 2
show_training = None
show_metrics = True

scaler_params, x_train, x_test = pbi.standard_scaler(x_train, x_test)

x_train, y_train = pbi.auto_balancer(x_train, y_train)
x_test, y_test = pbi.synthetic_augmentation(x_test, y_test)


W = pbi.fit(x_train, y_train, activation_potential, show_training)


test_model = pbi.evaluate(x_test, y_test, activation_potential, show_metrics, W)

test_preds = test_model[pbi.get_preds()]
test_acc = test_model[pbi.get_acc()]

model_name = 'creditcard_fraud'
model_type = 'PLAN'
weights_type = 'txt'
weights_format = 'd'
model_path = 'PlanModels'

pbi.save_model(model_name, model_type, class_count, activation_potential, test_acc, weights_type, weights_format, model_path, scaler_params, W)


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

precisison, recall, f1 = pbi.metrics(y_test, test_preds)

print('Precision: ', precisison, '\n', 'Recall: ', recall, '\n', 'F1: ', f1)
