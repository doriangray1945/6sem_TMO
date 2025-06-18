import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Заголовок приложения
st.title('Демонстрация моделей машинного обучения')
st.write("""Это приложение позволяет сравнивать разные модели регрессии на ваших данных.""")

# 1. Загрузка данных
st.sidebar.header('1. Загрузка данных')
uploaded_file = st.sidebar.file_uploader("Загрузите CSV файл", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader('Предпросмотр данных')
    st.write(data.head())
    
    # Выбор признаков
    st.sidebar.header('2. Выбор признаков')
    features = st.sidebar.multiselect('Выберите независимые переменные', data.columns)
    target = st.sidebar.selectbox('Выберите целевую переменную', data.columns)
    
    if features and target:
        X = data[features]
        y = data[target]
        
        # 2. Разделение данных
        test_size = st.sidebar.slider('Размер тестовой выборки (%)', 10, 40, 20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
        
        # 3. Выбор модели
        st.sidebar.header('3. Выбор модели')
        model_name = st.sidebar.selectbox('Выберите модель', 
                                        ['Линейная регрессия', 
                                         'Дерево решений', 
                                         'Случайный лес'])
        
        # Гиперпараметры моделей
        if model_name == 'Дерево решений':
            max_depth = st.sidebar.slider('Максимальная глубина', 1, 20, 5)
            model = DecisionTreeRegressor(max_depth=max_depth)
        elif model_name == 'Случайный лес':
            n_estimators = st.sidebar.slider('Количество деревьев', 10, 200, 100)
            max_depth = st.sidebar.slider('Максимальная глубина', 1, 20, 5)
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
        else:
            model = LinearRegression()
        
        # Обучение модели
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # 4. Результаты
        st.subheader('Результаты моделирования')
        st.write(f'**Модель:** {model_name}')
        
        # Метрики
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f'**MSE:** {mse:.2f}')
        st.write(f'**R2 Score:** {r2:.2f}')
        
        # График
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        ax.set_xlabel('Истинные значения')
        ax.set_ylabel('Предсказанные значения')
        ax.set_title('График предсказаний')
        st.pyplot(fig)
        
        # Важность признаков (для деревьев)
        if hasattr(model, 'feature_importances_'):
            st.subheader('Важность признаков')
            feat_importances = pd.Series(model.feature_importances_, index=features)
            fig2, ax2 = plt.subplots()
            feat_importances.plot.bar(ax=ax2)
            st.pyplot(fig2)
    else:
        st.warning('Пожалуйста, выберите признаки и целевую переменную')
else:
    st.info('Пожалуйста, загрузите CSV файл для анализа')