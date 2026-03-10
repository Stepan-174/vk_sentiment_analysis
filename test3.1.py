import vk_api
import json
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
import torch
from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Введите ваш токен API
TOKEN = 'c64'  # замените на свой токен

# Список ID сообществ (например, для групп: -123456789)
GROUP_IDS = [175005324, 32182751, 29700547, 87721351, 36959676]  # замените на нужные ID сообществ

# Загрузка NLTK ресурсов
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('russian'))

# Загрузка модели для оценки тональности (на русском)
model_name = 'blanchefort/rubert-base-cased-sentiment'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Функция очистки текста
def clean_text(text):
    # Удаление ссылок
    text = re.sub(r'http\S+', '', text)
    # Удаление эмодзи и спецсимволов
    text = re.sub(r'[^\w\s]', '', text)
    # Приведение к нижнему регистру
    text = text.lower()
    # Токенизация
    tokens = word_tokenize(text)
    # Удаление стоп-слов
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# Функция анализа тональности
def get_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    score, predicted_class = torch.max(probs, dim=1)
    sentiment_label = {0: 'negative', 1: 'neutral', 2: 'positive'}[predicted_class.item()]
    return sentiment_label

# Получение данных
def get_posts_and_comments(token, group_ids, posts_per_group=10, comments_per_post=5):
    vk = vk_api.VkApi(token=token).get_api()
    raw_data = []

    for group_id in group_ids:
        try:
            posts = vk.wall.get(owner_id=-group_id, count=posts_per_group)['items']
        except Exception as e:
            print(f"Ошибка при получении постов для сообщества {group_id}: {e}")
            continue

        for post in posts:
            post_id = post['id']
            date_unix = post.get('date', 0)
            date_str = datetime.fromtimestamp(date_unix).strftime('%Y-%m-%d %H:%M:%S')

            post_info = {
                'group_id': group_id,
                'post_id': post_id,
                'text': post.get('text', ''),
                'date': date_str,
                'likes': post.get('likes', {}).get('count', 0),
                'reposts': post.get('reposts', {}).get('count', 0),
                'views': post.get('views', {}).get('count', None),
                'comments_count': post.get('comments', {}).get('count', 0),
                'comments': []
            }

            try:
                comments = vk.wall.getComments(owner_id=-group_id, post_id=post_id, count=comments_per_post, sort='desc')
            except Exception as e:
                print(f"Ошибка при получении комментариев поста {post_id} в сообществе {group_id}: {e}")
                comments = {'items': []}

            for comment in comments.get('items', []):
                comment_date = datetime.fromtimestamp(comment.get('date', 0)).strftime('%Y-%m-%d %H:%M:%S')
                comment_info = {
                    'text': comment.get('text', ''),
                    'likes': comment.get('likes', {}).get('count', 0),
                    'date': comment_date
                }
                post_info['comments'].append(comment_info)

            raw_data.append(post_info)

    return raw_data

# Получение данных и сохранение оригинальных
raw_data = get_posts_and_comments(TOKEN, GROUP_IDS)

with open('discourse.txt', 'w', encoding='utf-8') as f:
    json.dump(raw_data, f, ensure_ascii=False, indent=4)

# Анализ и подготовка данных для вывода
sentiment_results = []

for post in raw_data:
    # Анализ поста
    original_text = post['text']
    cleaned_text = clean_text(original_text)
    sentiment_post = get_sentiment(cleaned_text)

    # Анализ комментариев
    comments_sentiments = []
    for comment in post['comments']:
        original_comment_text = comment['text']
        cleaned_comment_text = clean_text(original_comment_text)
        sentiment_comment = get_sentiment(cleaned_comment_text)
        comments_sentiments.append({
            'cleaned_text': cleaned_comment_text,
            'sentiment': sentiment_comment,
            'likes': comment['likes'],
            'date': comment['date']
        })

    sentiment_results.append({
	'group_id': post['group_id'],
        'post_id': post['post_id'],
        'cleaned_text': cleaned_text,
        'sentiment': sentiment_post,
        'date': post['date'],
        'likes': post['likes'],
        'reposts': post['reposts'],
        'views': post['views'],
	'comments_count': post['comments_count'],
        'comments_sentiments': comments_sentiments
    })

# Сохраняем результаты анализа
with open('sentiment_analysis.txt', 'w', encoding='utf-8') as f:
    json.dump(sentiment_results, f, ensure_ascii=False, indent=4)

print("Все данные обработаны и сохранены.")