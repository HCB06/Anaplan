import numpy as np
from anaplan import plan
import pickle

inp = ['very bad, too bad discusting!, like a shit, it is poor casting and poor film with a really bad and worst scenario']


with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

inp_vectorized = vectorizer.transform(inp)

inp = inp_vectorized.toarray()

# Model ile tahmin yapma
predict = plan.predict_model_ssd(Input=inp, model_name='IMDB', model_path='')

# Tahmini yorumlama
if np.argmax(predict) == 1:
    predict_label = 'pozitif'
elif np.argmax(predict) == 0:
    predict_label = 'negatif'

print('%' + str(int(max(plan.Softmax(predict) * 100))) + ' ' + predict_label)
