
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer



nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('tagsets/pt_core_web_sm')
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('tagsets/pt_core_web_sm')
nltk.download('averaged_perceptron_tagger')

ds = pd.read_csv('base_de_dados\Olympics_Tokyo_tweets.csv',low_memory=False)

sid = SentimentIntensityAnalyzer()

tweets = ds['text'].tolist()

Analise = {'Positivo': 0, 'Neutro': 0, 'Negativo': 0}

nome_arquivo_csv = "C:/Users/auric/Documents/Tcc/base_de_dados/Olympics_Tokyo_tweets.csv"



def extrair_assunto(frase):

    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(frase) 
    tags = pos_tag(tokens) 
    print(tags)
    subject = [ lemmatizer.lemmatize(word, pos='v') for word, tag in tags if tag.startswith("VB") or tag == "NN"] 
    print(subject)

    if len(subject) >= 1: 
      subject = subject[0]

    else:
      subject = None

    return subject

def processar_csv(nome_arquivo_csv):
    df = pd.read_csv(nome_arquivo_csv, low_memory=False)
    assuntos_principais = []

    for index, row in df.iterrows():
        texto = row['text']
        assunto_principal = extrair_assunto(texto)
        assuntos_principais.append(assunto_principal)

    return assuntos_principais


assuntos_principais = processar_csv(nome_arquivo_csv)

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

for i, assunto in enumerate(assuntos_principais):
    print(f"Linha {i+1}: {assunto}")


Analise_Resultado = list(Analise.keys())
Analise_Valor = list(Analise.values())

df = pd.DataFrame({
   'Tweets': tweets,
   'Assunto Principal': assuntos_principais
})

df['Sentimento Positivo'] = Analise['Positivo']
df['Sentimento Neutro'] = Analise['Neutro']
df['Sentimento Negativo'] = Analise['Negativo']
df.to_csv('analise_tweets.csv', index=False)


# Cria um gráfico de dispersão das polaridades e subjetividades
# plt.bar(Analise_Resultado, Analise_Valor, color = 'Maroon', width = 0.4)
# plt.title('Análise sentimental dos tweets')
# plt.xlabel('Classificação')
# plt.ylabel('Numero de resultados')
# plt.show()

# plt.pie(Analise_Valor, labels = Analise_Resultado)
# plt.show()





