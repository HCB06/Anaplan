import time
from colorama import Fore
from pyerualjetwork import plan, planeat, data_operations, model_operations, metrics
from sklearn.datasets import load_iris
import numpy as np

data = load_iris()
X = data.data
y = data.target

x_train, x_test, y_train, y_test = data_operations.split(X, y, 0.3, 42)

y_train, y_test = data_operations.encode_one_hot(y_train, y_test)

x_train, y_train = data_operations.synthetic_augmentation(x_train, y_train)

scaler_params, x_train, x_test = data_operations.standard_scaler(x_train, x_test)

# Configuring optimizator
genetic_optimizer = lambda *args, **kwargs: planeat.evolve(*args, activation_change_prob=0.3, activation_add_prob=0.9, policy='more_selective', **kwargs)

model = plan.learner(x_train, y_train, optimizator=genetic_optimizer, strategy='accuracy', neural_web_history=True, target_acc=0.88, interval=16.67) # learner function = TFL(Test Feedback Learning). If test parameters not given then uses Train Feedback. More information: https://github.com/HCB06/pyerualjetwork/blob/main/Welcome_to_plan/plan.pdf

W = model[model_operations.get_weights()]

test_model = plan.evaluate(x_test, y_test, W=W, activation_potentiation=model[model_operations.get_act_pot()])

test_preds = test_model[model_operations.get_preds()]
test_acc = test_model[model_operations.get_acc()]

model_operations.save_model(model_name='iris',
                 model_type='plan',
                 test_acc=test_acc,
                 weights_type='npy',
                 weights_format='raw',
                 model_path='',
                 activation_potentiation=model[model_operations.get_act_pot()],
                 scaler_params=scaler_params,
                 show_architecture=True,
                 W=W)

precisison, recall, f1 = metrics.metrics(y_test, test_preds)
print('Precision: ', precisison, '\n', 'Recall: ', recall, '\n', 'F1: ', f1)

y_test = data_operations.decode_one_hot(y_test)


for i in range(len(x_test)):
    Predict = model_operations.predict_model_ram(x_test[i], W=W, activation_potentiation=model[model_operations.get_act_pot()])
    time.sleep(0.6)
    if np.argmax(Predict) == y_test[i]:
        print(Fore.GREEN + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', y_test[i])
    else:
        print(Fore.RED + 'Predicted Output(index):', np.argmax(Predict), 'Real Output(index):', y_test[i])
