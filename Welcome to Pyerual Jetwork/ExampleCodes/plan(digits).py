import plan
from sklearn.datasets import load_digits

data = load_digits()

X = data.data
y = data.target

X = plan.normalization(X)
    
x_train, x_test, y_train, y_test = plan.split(X, y, 0.4, 42)


y_train, y_test = plan.encode_one_hot(y_train, y_test)


x_test, y_test = plan.auto_balancer(x_test, y_test)


W = plan.fit(x_train, y_train)

W = plan.normalization(W)

test_model = plan.evaluate(x_test, y_test, show_metrices=True, W=W)
test_preds = test_model[get_preds()]
test_acc = test_model[get_acc()]
