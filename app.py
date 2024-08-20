import os
import time
import telebot
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import io
import matplotlib

from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from openai import APIConnectionError

# Установить бэкенд для Matplotlib
matplotlib.use('Agg')

# Укажите ваш токен Telegram-бота
API_TOKEN = '7033028733:AAFTn3K4h9bAV0dEPSClYIMk_RPTlQaMUcg'
bot = telebot.TeleBot(API_TOKEN, parse_mode=None)

# Промокоды
PROMO_CODES = {"BEGZOD2024", "CRYPTO2024"}

# Словарь для хранения состояния пользователей
user_states = {}

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
    model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
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

# Создание кнопок
def create_buttons():
    markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = telebot.types.KeyboardButton('✅ Получить прогнозы')
    btn2 = telebot.types.KeyboardButton('📊 Показать график')
    btn3 = telebot.types.KeyboardButton('🔑 Ввести промокод')
    btn4 = telebot.types.KeyboardButton('⚙️ о функциях')
    btn5 = telebot.types.KeyboardButton('💬 Спросить у ИИ')
    markup.add(btn1, btn2, btn3, btn4, btn5)
    return markup

# Визуализация данных
def plot_predictions(df):
    plt.figure(figsize=(14, 8))
    df = df.sort_values(by='market_cap', ascending=False).head(10)
    colors = ['green' if pred == 1 else 'red' for pred in df['prediction']]
    bars = plt.bar(df['name'], df['current_price'], color=colors, edgecolor='black')

    for bar, pred in zip(bars, df['prediction']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height,
                 'Купить' if pred == 1 else 'Продать', ha='center', va='bottom', color='black')

    plt.xlabel('Криптовалюта', fontsize=14)
    plt.ylabel('Текущая цена (USD)', fontsize=14)
    plt.title('Топ-10 криптовалют по рыночной стоимости с прогнозами', fontsize=16)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Добавление сетки и улучшение внешнего вида
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.gca().set_axisbelow(True)

    # Дополнительная информация и рекомендации
    plt.figtext(0.99, 0.01, 'Зеленые столбики означают сигнал на покупку, красные - на продажу.', horizontalalignment='right')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    return buf

# Класс для интеграции LLM
class LLMChatBot:
    def __init__(self):
        self.llm = ChatOpenAI(
            temperature=0.95,
            model="glm-4",
            openai_api_key="Your api",
            openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
        )
        self.prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    "You are a nice chatbot having a conversation with a human and you can answer in Russian language"
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{question}")
            ]
        )
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.conversation = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            verbose=True,
            memory=self.memory
        )

    def ask_question(self, question):
        try:
            response = self.conversation.invoke({"question": question})
            return response['text']
        except APIConnectionError as err:
            return "Произошла ошибка соединения с API. Пожалуйста, попробуйте позже."

# Инициализация LLMChatBot
llm_chatbot = LLMChatBot()

@bot.message_handler(commands=['start'])
def send_welcome(message):
    markup = create_buttons()
    bot.send_message(message.chat.id, "Добро пожаловать! Пожалуйста, активируйте свой аккаунт с помощью промокода для получения доступа к функциям.", reply_markup=markup)
    user_states[message.chat.id] = {"activated": False}

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_state = user_states.get(message.chat.id, {"activated": False})

    if message.text == '✅ Получить прогнозы':
        if user_state["activated"]:
            bot.send_message(message.chat.id, "Получение данных и прогнозирование ⏳...")
            df = get_crypto_data()
            model = train_model(df)
            predictions = predict(model, df)

            top_predictions = predictions[['name', 'current_price', 'prediction']].head(10)
            response = "Топ-10 криптовалютных прогнозов:\n\n"
            for index, row in top_predictions.iterrows():
                trend = "вверх📈 " if row['prediction'] == 1 else "вниз📉"
                response += f"{row['name']}: ${row['current_price']} - Предсказание: {trend}\n"

            bot.send_message(message.chat.id, response)
        else:
            bot.send_message(message.chat.id, "Пожалуйста, активируйте свой аккаунт с помощью промокода.")

    elif message.text == '📊 Показать график':
        if user_state["activated"]:
            df = get_crypto_data()
            model = train_model(df)
            predictions = predict(model, df)
            buf = plot_predictions(predictions)
            bot.send_photo(message.chat.id, buf)
        else:
            bot.send_message(message.chat.id, "Пожалуйста, активируйте свой аккаунт с помощью промокода.")

    elif message.text == '🔑 Ввести промокод':
        bot.send_message(message.chat.id, "Пожалуйста, введите ваш промокод.")
        user_states[message.chat.id]['awaiting_promo'] = True

    elif message.text == '⚙️ о функциях':
        bot.send_message(message.chat.id, "Функционал бота: Этот Telegram-бот предоставляет пользователям прогнозы по криптовалютам и визуализацию данных с помощью графиков. Основные функции бота включают:\n\n"
                                          "1. Получение прогнозов:\n"
                                          "- Команда: ✅ Получить прогнозы\n"
                                          "- Описание: Бот получает данные о текущих ценах криптовалют, прогнозирует их будущее изменение и предоставляет пользователю топ-10 прогнозов.\n\n"
                                          "2. Показать график:\n"
                                          "- Команда: 📊 Показать график\n"
                                          "- Описание: Бот визуализирует данные о криптовалютах, показывая текущие цены и прогнозы в виде графика.\n\n"
                                          "3. Ввести промокод:\n"
                                          "- Команда: 🔑 Ввести промокод\n"
                                          "- Описание: Пользователь может ввести промокод для активации дополнительных функций бота.\n\n"
                                          "4. Информация о функциях:\n"
                                          "- Команда: ⚙️ о функциях\n"
                                          "- Описание: Бот предоставляет информацию о всех доступных функциях и их использовании.\n\n"
                                          "5. Спросить у ИИ:\n"
                                          "- Команда: 💬 Спросить у ИИ\n"
                                          "- Описание: Бот отвечает на вопросы пользователя на основе ИИ-модели.")

    elif message.text == '💬 Спросить у ИИ':
      if user_state["activated"]:
        bot.send_message(message.chat.id, "Пожалуйста, введите ваш вопрос.")
        user_states[message.chat.id]['awaiting_ai_question'] = True
      else:
        bot.send_message(message.chat.id, "Пожалуйста, активируйте свой аккаунт с помощью промокода.")

    elif user_state.get('awaiting_promo', False):
        promo_code = message.text.strip().upper()
        if promo_code in PROMO_CODES:
            user_states[message.chat.id]['activated'] = True
            bot.send_message(message.chat.id, "Ваш промокод принят! Ваш аккаунт активирован.")
        else:
            bot.send_message(message.chat.id, "Неверный промокод. Пожалуйста, попробуйте снова.")
        user_states[message.chat.id]['awaiting_promo'] = False

    elif user_state.get('awaiting_ai_question', False):
        question = message.text.strip()
        bot.send_message(message.chat.id, "Пожалуйста, подождите, ваш вопрос обрабатывается... ⏳")
        response = llm_chatbot.ask_question(question)
        bot.send_message(message.chat.id, response)
        user_states[message.chat.id]['awaiting_ai_question'] = False

    else:
        bot.send_message(message.chat.id, "Неизвестная команда. Пожалуйста, выберите одну из доступных команд.")

while True:
    try:
        bot.polling(non_stop=True, interval=1, timeout=60)
    except Exception as e:
        print(f"Ошибка: {e}")
        time.sleep(15)
