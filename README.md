## About
Модель машинного обучение, обученная на [наборе данных](https://www.kaggle.com/datasets/blackmoon/russian-language-toxic-comments), умеющая определять токсичные комментарии на русском языке.

Для построение выбора итоговой модели я сравнил несколько алгоритмов с разных библиотек и лучше все себя показал алгоритм из библиотеки CatBoost (CatboostClassifier)

Подробнее о наборе данных - [тут](https://habr.com/ru/companies/vk/articles/526268/).

Для обертки решения я использовал framework - [FastApi](https://fastapi.tiangolo.com) для создания простого Rest API, а так же обернул решение в Docker контейнер.

Для обработке входящего комментария был написал класс text_preproccesing.py, состоящий из:
- Функция удаления символов и стоп слов.
- Функция преобразования слова к нормальной форме.
- Функция получение текста в векторном представление с использованием пред. обученной модели [RuBert](https://habr.com/ru/articles/562064/) из библиотеки [PyTorch](https://pytorch.org)

## How to run
1. Так как программа обернута в докер контейнер для начала нужно его скачать, в терминал введите команду:
```bash
docker image pull justpain/neural_api:v1.0.1
```
2. Далее вам нужно запустить скачанный образ командой:
```bash
docker run -d -p 7338:8000 justpain/neural_api:v1.0.1
```
Вместо порта 7338 можете указать любой.