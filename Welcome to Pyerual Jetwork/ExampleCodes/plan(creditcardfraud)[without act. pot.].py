import pandas as pd
import plan

data = pd.read_csv('creditcard.csv')

x = data.drop('Class', axis=1).values
y = data['Class'].values

x = plan.normalization(x)
x_train, x_test, y_train, y_test = plan.split(x, y, 0.4, 42)

y_train, y_test = plan.encode_one_hot(y_train, y_test)


scaler_params, x_train, x_test = plan.standard_scaler(x_train, x_test)

x_train, y_train = plan.auto_balancer(x_train, y_train)
x_test, y_test = plan.auto_balancer(x_test, y_test)


W = plan.fit(x_train, y_train)


test_model = plan.evaluate(x_test, y_test, show_metrices=True, W=W)
