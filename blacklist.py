import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Ваш исходный датасет
data = {'Age': [14, 34, 42, 30, 16],
        'Gender': ['male', 'female', 'male', 'male', 'male'],
        'Region': ['city', 'city', 'countryside', 'countryside', 'city'],
        'Occupation': ['student', 'teacher', 'banker', 'teacher', 'student'],
        'Income': [0, 22000, 24000, 25000, 0],
        'Has Laptop': ['no', 'no', 'yes', 'no', 'no']}

df = pd.DataFrame(data)

# Разделяем на признаки (X) и целевую переменную (y)
X = df.drop('Has Laptop', axis=1)
y = df['Has Laptop']

# Применяем факторизацию от pandas
X_encoded = pd.get_dummies(X, columns=['Gender', 'Region', 'Occupation'])

# Разбиваем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Создаем модель дерева решений
model = DecisionTreeClassifier(random_state=42)

# Обучаем модель
model.fit(X_train, y_train)

# Оцениваем точность модели
accuracy = model.score(X_test, y_test)
print(f'Accuracy: {accuracy}')
