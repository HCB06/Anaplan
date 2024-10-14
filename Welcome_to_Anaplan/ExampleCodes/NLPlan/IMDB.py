import numpy as np
from anaplan import plan
import pickle

with open('tfidf_imdb.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

model = plan.load_model(model_name='IMDB', model_path='')

while True:

    text = input()

    if text == '':

        text = None

    try:

        text_news = [text]

        inp_vectorized = vectorizer.transform(text_news)

        inp = inp_vectorized.toarray()

        # Model ile tahmin yapma
        predict = plan.predict_model_ram(Input=inp, W=model[plan.get_weights()], activation_potentiation=model[plan.get_act_pot()], scaler_params=model[plan.get_scaler()])


        # Tahmini yorumlama
        if np.argmax(predict) == 1:
            predict_label = 'pozitif'
        elif np.argmax(predict) == 0:
            predict_label = 'negatif'

        print('%' + str(int(max(plan.Softmax(predict) * 100))) + ' ' + predict_label + '\n')

    except:

        pass