import json
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from keybert import KeyBERT

# Константы
BASE_URL = "https://vk.com/wall"

# Загрузка данных
with open('sentiment_analysis.txt', 'r', encoding='utf-8') as f:
    posts = json.load(f)

# Общее число постов
total_posts = len(posts)

# Собираем все комментарии
all_comments_texts = [
    c['cleaned_text']
    for p in posts
    for c in p.get('comments_sentiments', [])
]

# Анализ тональностей постов
post_sentiments = [p['sentiment'] for p in posts]
post_counts = Counter(post_sentiments)

# Анализ тональностей комментариев
comments_sentiments = [
    c['sentiment']
    for p in posts
    for c in p.get('comments_sentiments', [])
]
total_comments = len(comments_sentiments)
comments_counts = Counter(comments_sentiments)

# Процентное распределение
def percent(count, total):
    return (count / total * 100) if total else 0

# Процент по тональностям
post_percentages = {k: percent(v, total_posts) for k, v in post_counts.items()}
comments_percentages = {k: percent(v, total_comments) for k, v in comments_counts.items()}

# Сравнение тональностей постов и комментариев
match_count = 0
for p in posts:
    p_sentiment = p['sentiment']
    for c in p.get('comments_sentiments', []):
        if c['sentiment'] == p_sentiment:
            match_count +=1
match_ratio = percent(match_count, total_comments)

# Интерпретация
if match_ratio > 70:
    interpretation = "Тональности постов и комментариев в основном совпадают."
elif match_ratio > 40:
    interpretation = "Отношение тональностей постов и комментариев неоднозначное."
else:
    interpretation = "Тональности постов и комментариев значительно не совпадают."

# Анализ ключевых тем с помощью KeyBERT
if all_comments_texts:
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(" ".join(all_comments_texts), top_n=10)
    key_terms = ", ".join([k[0] for k in keywords])
else:
    key_terms = "Темы не выделены (нет комментариев)."

# Находим 10 самых негативных постов по количеству комментариев
negative_posts = [p for p in posts if p['sentiment'] == 'negative']
top_negative_posts = sorted(negative_posts, key=lambda p: p.get('comments_count', 0), reverse=True)[:10]

# Создаем DataFrame для графиков
df_posts = pd.DataFrame(posts)
# Добавим колонку с длиной текста
df_posts['text_length'] = df_posts['cleaned_text'].apply(lambda x: len(x.split()))
# Distribution of sentiments
sentiment_order = ['positive', 'neutral', 'negative']

# Построение графиков
plt.figure(figsize=(12,6))
# Распределение постов по тональности
plt.subplot(1,2,1)
sns.countplot(data=df_posts, x='sentiment', order=sentiment_order)
plt.title('Распределение постов по тональности')

# Распределение комментариев по тональности
comments_df = pd.DataFrame(comments_sentiments, columns=['sentiment'])
plt.subplot(1,2,2)
sns.countplot(data=comments_df, x='sentiment', order=sentiment_order)
plt.title('Распределение комментариев по тональности')

plt.tight_layout()
plt.savefig('visualization.png')

# Формируем текст сводки
lines = []

lines.append("ИТОГОВЫЙ АНАЛИЗ\n")
lines.append(f"Общее количество постов: {total_posts}")
lines.append(f"Общее количество комментариев: {total_comments}\n")

lines.append("Распределение тональностей постов:")
for s in sentiment_order:
    lines.append(f" - {s.capitalize()}: {post_counts.get(s,0)} ({post_percentages.get(s,0):.2f}%)")
lines.append("\nРаспределение тональностей комментариев:")
for s in sentiment_order:
    lines.append(f" - {s.capitalize()}: {comments_counts.get(s,0)} ({comments_percentages.get(s,0):.2f}%)")

lines.append(f"\nПроцент совпадения тональностей постов и комментариев: {match_ratio:.2f}%")
lines.append(interpretation)

lines.append("\nКлючевые темы обсуждений (по KeyBERT):")
lines.append(key_terms)

lines.append("\n10 самых негативных постов по количеству комментариев:\n")
for idx, p in enumerate(top_negative_posts, 1):
    group_id = p['group_id']
    post_id = p['post_id']
    text = p['cleaned_text']
    sentiment = p['sentiment']
    comments_cnt = p.get('comments_count', 0)
    url = f"{BASE_URL}{-group_id}_{post_id}"
    lines.append(f"{idx}. Текст: {text}")
    lines.append(f"   Тональность: {sentiment}")
    lines.append(f"   ID сообщества: {group_id}")
    lines.append(f"   ID поста: {post_id}")
    lines.append(f"   URL: {url}")
    lines.append(f"   Количество комментариев: {comments_cnt}\n")

# Сохраняем итог
with open('resume.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(lines))

print("Анализ завершен! Итоговая сводка сохранена в файл resume.txt, а графики — в visualization.png.")