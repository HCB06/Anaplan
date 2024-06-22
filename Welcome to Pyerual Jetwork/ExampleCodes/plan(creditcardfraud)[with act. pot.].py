import pandas as pd
import plan

data = pd.read_csv('creditcard.csv')

x = data.drop('Class', axis=1).values
y = data['Class'].values


x_train, x_test, y_train, y_test = plan.split(x, y, 0.4, 42)

y_train, y_test = plan.encode_one_hot(y_train, y_test)

x_train = x_train.tolist()
x_test = x_test.tolist()

activation_potentiation = 0
class_count = 2
show_training = None
show_metrics = True

scaler_params, x_train, x_test = plan.standard_scaler(x_train, x_test)

x_train, y_train = plan.auto_balancer(x_train, y_train)
x_test, y_test = plan.synthetic_augmentation(x_test, y_test)


W = plan.fit(x_train, y_train, show_training, activation_potentiation)


test_model = plan.evaluate(x_test, y_test, show_metrics, W, activation_potentiation)

test_preds = test_model[plan.get_preds()]
test_acc = test_model[plan.get_acc()]

model_name = 'creditcard_fraud'
model_type = 'PLAN'
weights_type = 'txt'
weights_format = 'd'
model_path = 'PlanModels'

plan.save_model(model_name, model_type, class_count, test_acc, weights_type, weights_format, model_path, scaler_params, W, activation_potentiation)


"""
y_test = np.argmax(y_test, axis=1)
scaler_params = None
for i in range(len(x_test)):
    Predict = plan.predict_model_ssd(x_test[i], model_name, model_path)
    time.sleep(0.6)
    if np.argmax(Predict) == y_test[i]:
        print(Fore.GREEN + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', y_test[i])
    else:
        print(Fore.RED + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', y_test[i])

"""

precisison, recall, f1 = plan.metrics(y_test, test_preds)

print('Precision: ', precisison, '\n', 'Recall: ', recall, '\n', 'F1: ', f1)
