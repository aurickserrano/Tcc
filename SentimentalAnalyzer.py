# Consolidando todo o código em uma célula para fácil referência

# Importações
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Carregar dados
# df = pd.read_csv('/mnt/data/Olympics_Tokyo_tweets.csv')  # Já carregado anteriormente

# Tratamento de texto (remoção de caracteres especiais e emoticons)
def clean_text(text):
    import re
    return re.sub(r'[^\w\s]', '', text)

# df['cleaned_text'] = df['text'].apply(clean_text)  # Já feito anteriormente

# Classificação por categoria (em inglês)
english_keywords = {
    'Opening_Ceremony': ['opening', 'ceremony', 'start', 'ateez', 'ateezofficial'],
    'Sports_Events': ['athletics', 'swimming', 'gymnastics', 'competition', 'event',
                      'archery', 'artistic gymnastics', 'artistic swimming', 'athletics', 'badminton',
                      'baseball', 'softball', 'basketball', 'beach volleyball', 'boxing', 'canoe slalom',
                      'canoe sprint', 'cycling bmx freestyle', 'cycling bmx racing', 'cycling mountain bike',
                      'cycling road', 'cycling track', 'diving', 'equestrian', 'fencing', 'football', 'golf',
                      'handball', 'hockey', 'judo', 'karate', 'marathon swimming', 'modern pentathlon',
                      'rhythmic gymnastics', 'rowing', 'rugby sevens', 'sailing', 'shooting', 'skateboarding',
                      'sport climbing', 'surfing', 'swimming', 'table tennis', 'taekwondo', 'tennis',
                      'trampoline gymnastics', 'triathlon', 'volleyball', 'water polo', 'weightlifting', 'wrestling'],
    'Results_and_Medals': ['medal', 'gold', 'silver', 'bronze', 'winner'],
    'Controversies_and_Scandals': ['controversy', 'scandal', 'doping'],
    'Media_Coverage': ['media', 'coverage', 'news']
}

def classify_tweet_english(tweet):
    for category, keys in english_keywords.items():
        if any(key in tweet.lower() for key in keys):
            return category
    return 'Others'

# df['category_english_updated'] = df['cleaned_text'].apply(classify_tweet_english)  # Já feito anteriormente

# Análise de sentimento com limites ajustados
def adjusted_sentiment_analysis(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0.2:
        return 'Positive'
    elif polarity < -0.2:
        return 'Negative'
    else:
        return 'Neutral'

# Amostra de 51% dos dados
df_sample_51 = df_cleaned.sample(frac=0.51, random_state=42)

# Aplicar análise de sentimento
df_sample_51['adjusted_sentiment'] = df_sample_51['cleaned_text'].apply(adjusted_sentiment_analysis)

# Agrupar por categoria e sentimento
adjusted_sentiment_by_category_51 = df_sample_51.groupby(['category_english_updated', 'adjusted_sentiment']).size().reset_index(name='count')

# Gráficos
categories = adjusted_sentiment_by_category_51['category_english_updated'].unique()
for category in categories:
    plt.figure(figsize=(10, 6))
    sns.barplot(x='adjusted_sentiment', y='count', data=adjusted_sentiment_by_category_51[adjusted_sentiment_by_category_51['category_english_updated'] == category])
    plt.title(f'Adjusted Sentiment Analysis for {category} (51% Sample)')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()
