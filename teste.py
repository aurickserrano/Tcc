import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Baxiando as bibloetcas que iremos utilizar
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('tagsets/pt_core_web_sm')
nltk.download('averaged_perceptron_tagger')

sid = SentimentIntensityAnalyzer()

# Cria uma lista para armazenar as frases filtradas
ds = pd.read_csv('base_de_dados\Olympics_Tokyo_tweets.csv' ,low_memory=False )
tweets = ds['text'].tolist()

# Cria listas para armazenar as polaridades e subjetividades
Analise = {'Positivo': 0, 'Neutro': 0, 'Negativo': 0}

# Realiza a análise sentimental dos tweets e adiciona as polaridades e subjetividades às listas
for tweet in tweets:
    texto = tweet
    resultado = sid.polarity_scores(texto)
    polaridade = resultado['compound']
    if polaridade > 0.5:
      Analise['Positivo'] += 1
    elif polaridade > -0.5 and polaridade < 0.5:
      Analise['Neutro'] += 1
    else:
      Analise['Negativo'] += 1

Analise_Resultado = list(Analise.keys())
Analise_Valor = list(Analise.values())

# Cria um gráfico de dispersão das polaridades e subjetividades
plt.bar(Analise_Resultado, Analise_Valor, color = 'Maroon', width = 0.4)
plt.title('Análise sentimental dos tweets')
plt.xlabel('Classificação')
plt.ylabel('Numero de resultados')
plt.show()

plt.pie(Analise_Valor, labels = Analise_Resultado)
plt.show()