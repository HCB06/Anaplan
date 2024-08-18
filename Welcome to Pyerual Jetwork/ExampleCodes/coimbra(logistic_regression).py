import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import plan

# Veriyi yükleme
file_path = 'breast_cancer_coimbra.csv'
data = pd.read_csv(file_path)


X = data.drop('Classification', axis=1).values
y = data['Classification'].values

# Eğitim, test ve doğrulama verilerini ayırma
x_train, x_test, y_train, y_test = plan.split(X, y, 0.4, 42)
x_train, x_val, y_train, y_val = plan.split(x_train, y_train, 0.2, 42)

y_val = plan.encode_one_hot(y_val, y_test)[0]
y_train, y_test = plan.encode_one_hot(y_train, y_test)

# Veri dengesizliği durumunda otomatik dengeleme
x_train, y_train = plan.auto_balancer(x_train, y_train)

y_train = plan.decode_one_hot(y_train)
y_test = plan.decode_one_hot(y_test)

# Verilerin standardize edilmesi
scaler_params, x_train, x_test = plan.standard_scaler(x_train, x_test)

# Lojistik Regresyon modelini tanımlama ve eğitme
model = LogisticRegression(random_state=42)
model.fit(x_train, y_train)

# Modeli test etme
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
test_acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_acc:.4f}")

x_val = plan.standard_scaler(x_test=x_val, scaler_params=scaler_params)

# Model tahminlerini değerlendirme
y_val_pred = model.predict(x_val)
for i in range(len(x_val)):
    print(f"Predicted Output: {y_val_pred[i]}, Real Output: {y_val[i]}")
