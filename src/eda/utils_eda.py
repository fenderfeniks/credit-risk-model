import pandas as pd


def collapse_rare_categories(df, columns, top_n=10):
    df_copy = df.copy()
    for col in columns:
        # Находим топ популярных значений
        top_values = df_copy[col].value_counts().index[:top_n]
        # Все, что не вошло в топ, заменяем на 'other'
        df_copy[col] = df_copy[col].where(df_copy[col].isin(top_values), 'other')
    return df_copy