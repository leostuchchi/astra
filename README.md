Проект magic_profile является частью проекта personal_assistant состоящего из четырех проектов (с отдельными репозиториями):

astra: подготовка натальных карт, психоматриц, расчета биоритмов и рекоммендаций на один день

testing: психологическое тестирование и потребностей пользователя

assistant: на основании расчетов astra и testing выдает персонализированные рекоммендации на один день на базе ollama

astra: развернут локально (включая бд), взаимодействие телеграм бот 

расчеты: сбор первичной информации пользователя, расчет натальной карты, психоматрицы, биоритмов, лунных фаз, расчет данных на день.

проект magic_profile: создание и запись в базу данных натального и психоматричного профиля пользователя

структура проекта magic_profile (для переработки):

docker-compose.yml

bot: 
config.py
handlers.py
__init__.py
main.py

backend: 
__init__.py
assistant.py
biorhythm_calculator.py
biorhythm_services.py
chart_services.py
database.py
db_connection.py
matrix_services.py
moon.py
natal_chart.py
predictions.py
prediction_services.py
psyho_matrix.py
user_services.py
