# vk_sentiment_analysis
Cистема сентимент-анализа информационного поля в социальной сети ВКонтакте

Проект выполнен на Python.

Требуются библиотеки: vk_api nltk, transformers, torch, numpy, pandas, matplotlib, seaborn scikit-learn, keybert.

Загрузить через командную строку командой: pip install vk_api nltk transformers torch numpy pandas matplotlib seaborn scikit-learn keybert

Так же необходимо получить токен VK API

Файлы в директории:

Запуск в Windows в командной строке: python test3.1.py & python test7.1.py

test3.1.py – содержит скрипт, который запускает сбор, очистку и предварительный сентимент-анализ текста и записывает данные в файлы discourse.txt и sentiment_analysis.txt.

ВНИМАНИЕ: в  файле test3.1.py TOKEN = 'c64'  # нужно заменить на свой токен. Иначе работать не будет.

test7.1.py – содержит скрипт, который запускает создание аналитического отчета и визуализации, а затем сохраняет полученные результаты в файлы resume.txt и visualization.png.

discourse.txt – содержит необработанный текст, собранный из постов и комментариев в формате JSON.

sentiment_analysis.txt - содержит обработанный текст, собранный из постов и комментариев с предварительно обработанным сентимент-анализом в формате JSON.

resume.txt – содержит отчёт по проведённому анализу тональности.

visualization.png – содержит визуализацию по проведённому анализу.

