import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import streamlit as st
import joblib
import io

# Важная настройка для корректной настройки pipeline!
import sklearn
sklearn.set_config(transform_output="pandas")

# Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler, OrdinalEncoder, QuantileTransformer, PowerTransformer, MaxAbsScaler
from sklearn.model_selection import GridSearchCV, KFold
from category_encoders import TargetEncoder
from sklearn.compose import make_column_selector, TransformedTargetRegressor

# for model learning
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score

#models
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.svm import SVC
from catboost import CatBoostRegressor
# import xgboost as xgb
# from xgboost import XGBRegressor

# Metrics
# from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
# Игнорировать все предупреждения
warnings.simplefilter(action='ignore', category=Warning)

# tunning hyperparamters model
import optuna
from optuna.samplers import TPESampler



def load_model():
    with open('model.pkl', 'rb') as file:
        model = joblib.load(file)

    return model

def main():
    st.title('Оценка точности работы модели')

    # Загрузка модели
    model = load_model()

    # Загрузка данных от пользователя
    uploaded_file = st.file_uploader("Загрузите свой CSV файл", type=['csv'])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Получение предсказаний от модели
        predictions = np.expm1(model.predict(data))

        # Вывод предсказаний
        st.write("Предсказания модели:")
        st.write(predictions)

        # Вычисление метрик
        actual_values = data['SalePrice']
        mse = mean_squared_error(actual_values, predictions)
        mae = mean_absolute_error(actual_values, predictions)
        r2 = r2_score(actual_values, predictions)

        # Вывод метрик
        st.write("Метрики модели:")
        st.write(f"Mean Squared Error: {mse}")
        st.write(f"Mean Absolute Error: {mae}")
        st.write(f"R^2 Score: {r2}")

if __name__ == '__main__':
    main()