from pyerualjetwork import plan, data_operations, model_operations
import numpy as np
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt

# TRAIN

data = load_digits()

X = data.data
y = data.target

X = plan.normalization(X)

x_train, x_test, y_train, y_test = data_operations.split(X, y, 0.4, 42)

y_train, y_test = data_operations.encode_one_hot(y_train, y_test)
x_test, y_test = data_operations.auto_balancer(x_test, y_test)

W = plan.fit(x_train, y_train)

# TEST

test_model = plan.evaluate(x_test, y_test, W=W)

model_operations.save_model(model_name='digits', W=W)

# REVERSE_PREDICT

output = [0,0,0,0,0,0,0,0,1,0]
Input = model_operations.reverse_predict_model_ram(output, W)

plt.imshow(np.reshape(Input, (8,8)))
plt.show()