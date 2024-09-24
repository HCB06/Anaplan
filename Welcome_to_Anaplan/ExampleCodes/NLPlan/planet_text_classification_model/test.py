import numpy as np
from anaplan import plan
import pickle

text = "I recently bought a used Toyota Camry, and I have to say, it’s been a fantastic choice so far! The reliability of the Toyota brand has always appealed to me, and this model has lived up to the hype. It has a smooth ride, great fuel efficiency, and enough room for my family. However, I've been doing some reading about regular maintenance schedules, and I want to ensure I keep it in top shape. I know that regular oil changes are crucial, but I'm also curious about other preventative measures I should take. What are the most important maintenance tasks for a Camry owner? Should I consider getting the transmission fluid changed even if it seems to be functioning fine? Also, I’m considering upgrading the sound system; what recommendations do you have for aftermarket systems that work well with this model? Any tips from fellow Camry owners would be greatly appreciated!"


text_imdb = [text]
text_news = [text]

with open('tfidf_imdb.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

inp_vectorized = vectorizer.transform(text_imdb)

inp = inp_vectorized.toarray()



# Model ile tahmin yapma
predict = plan.predict_model_ssd(Input=inp, model_name='IMDB', model_path='')

# Tahmini yorumlama
if np.argmax(predict) == 1:
    predict_label = 'pozitif'
elif np.argmax(predict) == 0:
    predict_label = 'negatif'

print('%' + str(int(max(plan.Softmax(predict) * 100))) + ' ' + predict_label)


with open('tfidf_20news.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

inp_vectorized = vectorizer.transform(text_news)

inp = inp_vectorized.toarray()

# Model ile tahmin yapma
predict = plan.predict_model_ssd(Input=inp, model_name='20newsgroup', model_path='')


# Tahmini yorumlama
if np.argmax(predict) == 0:
    predict_label = 'Alternatif Ateizm'
elif np.argmax(predict) == 1:
    predict_label = 'Bilgisayar Grafikleri'
elif np.argmax(predict) == 2:
    predict_label = 'Windows İşletim Sistemi'
elif np.argmax(predict) == 3:
    predict_label = 'IBM PC Donanımı'
elif np.argmax(predict) == 4:
    predict_label = 'Mac Donanımı'
elif np.argmax(predict) == 5:
    predict_label = 'X Penceresi Sistemi'
elif np.argmax(predict) == 6:
    predict_label = 'Satış İlanları'
elif np.argmax(predict) == 7:
    predict_label = 'Arabalar'
elif np.argmax(predict) == 8:
    predict_label = 'Motosikletler'
elif np.argmax(predict) == 9:
    predict_label = 'Baseball Sporları'
elif np.argmax(predict) == 10:
    predict_label = 'Hokey Sporları'
elif np.argmax(predict) == 11:
    predict_label = 'Kriptografi'
elif np.argmax(predict) == 12:
    predict_label = 'Elektronik Bilimi'
elif np.argmax(predict) == 13:
    predict_label = 'Tıp Bilimi'
elif np.argmax(predict) == 14:
    predict_label = 'Uzay Bilimi'
elif np.argmax(predict) == 15:
    predict_label = 'Hristiyanlık'
elif np.argmax(predict) == 16:
    predict_label = 'Silah Politikaları'
elif np.argmax(predict) == 17:
    predict_label = 'Orta Doğu Siyaseti'
elif np.argmax(predict) == 18:
    predict_label = 'Genel Siyaset Tartışmaları'
elif np.argmax(predict) == 19:
    predict_label = 'Din ve Diğer Tartışmalar'
    

print('%' + str(int(max(plan.Softmax(predict) * 100))) + ' ' + predict_label)