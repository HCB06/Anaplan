from anaplan import plan
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('spam_dataset.csv')


X = df['message_content']
y = df['is_spam']


vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(X)

X = X.toarray()

# Veriyi eğitim ve test setlerine ayırma
x_train, x_test, y_train, y_test = plan.split(X, y, 0.4, 42)

# One-hot encoding
y_train, y_test = plan.encode_one_hot(y_train, y_test)

# Veri dengeleme
x_train, y_train = plan.synthetic_augmentation(x_train, y_train)

# Ölçekleme
scaler_params, x_train, x_test = plan.standard_scaler(x_train, x_test)

# PLAN Modeli
model = plan.learner(x_train, y_train, x_test, y_test, neurons_history=True, target_acc=1, except_this=['circular']) # learner function = TFL(Test Feedback Learning). If test parameters not given then uses Train Feedback. More information: https://github.com/HCB06/Anaplan/blob/main/Welcome_to_PLAN/PLAN.pdf

# Modeli test etme
test_model = plan.evaluate(x_test, y_test, W=model[plan.get_weights()], show_metrics=True, activation_potentiation=model[plan.get_act_pot()])
