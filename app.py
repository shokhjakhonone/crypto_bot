import telebot
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Укажите ваш токен Telegram-бота
API_TOKEN = 'token from bot Father'
bot = telebot.TeleBot(API_TOKEN)

# Функция для получения данных о криптовалютах
def get_crypto_data():
    url = 'https://api.coingecko.com/api/v3/coins/markets'
    params = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': 100,
        'page': 1,
        'sparkline': 'false'
    }
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data)
    return df

# Обучение модели машинного обучения
def train_model(df):
    df['price_change_percentage_24h'] = df['price_change_percentage_24h'].fillna(0)
    df['target'] = df['price_change_percentage_24h'].apply(lambda x: 1 if x > 0 else 0)
    
    features = ['current_price', 'market_cap', 'total_volume']
    X = df[features]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model accuracy: {accuracy}')
    
    return model

# Функция для предсказания
def predict(model, df):
    features = ['current_price', 'market_cap', 'total_volume']
    X = df[features]
    predictions = model.predict(X)
    df['prediction'] = predictions
    return df

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Welcome to the Crypto Analysis Bot! Type /predict to get predictions.")

@bot.message_handler(commands=['predict'])
def send_predictions(message):
    bot.reply_to(message, "Getting data and making predictions...")
    df = get_crypto_data()
    model = train_model(df)
    predictions = predict(model, df)
    
    top_predictions = predictions[['name', 'current_price', 'prediction']].head(10)
    response = "Top 10 cryptocurrency predictions:\n\n"
    for index, row in top_predictions.iterrows():
        trend = "up" if row['prediction'] == 1 else "down"
        response += f"{row['name']}: ${row['current_price']} - Prediction: {trend}\n"
    
    bot.reply_to(message, response)

bot.polling()
