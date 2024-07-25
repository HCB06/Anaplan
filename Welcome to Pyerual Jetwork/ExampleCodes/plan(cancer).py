import plan
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data
y = data.target

x_train, x_test, y_train, y_test = plan.split(X, y, 0.4, 42)

scaler_params, x_train, x_test = plan.standard_scaler(x_train, x_test)

y_train, y_test = plan.encode_one_hot(y_train, y_test)

x_test, y_test = plan.auto_balancer(x_test, y_test)

W = plan.fit(x_train, y_train)

test_model = plan.evaluate(x_test, y_test, show_metrices=True, W=W)

test_preds = test_model[plan.get_preds()]
test_acc = test_model[plan.get_acc()]

model_name = 'breast_cancer'
model_type = 'PLAN'
weights_type = 'txt'
weights_format = 'f'
model_path = ''
class_count = 2


scaler_params = None # because x_test is already scaled. If you change this row model make wrong predicts for predect from ssd function. if you just want save model then delete this row. If you want prediction from x_test, don't change.

plan.save_model(model_name, model_type, class_count, test_acc, weights_type, weights_format, model_path, scaler_params, W)


precisison, recall, f1 = plan.metrics(y_test, test_preds)
print('Precision: ', precisison, '\n', 'Recall: ', recall, '\n', 'F1: ', f1)
