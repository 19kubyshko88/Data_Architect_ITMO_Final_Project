import numpy as np
import pandas as pd


def get_hor_vol_table(target: pd.core.series.Series, source: pd.core.series.Series):
    """
    Возвращает список ценовых уровней из таблицы горизонтальных объемов для целевой таблицы.
    В первой ячейке цена самая поздняя по времени

    target - Series для записи
    source - MultyIndex Series (DATETIME, close-интервал) - тиковый
    """

    price_intervals = source.index.get_level_values(1)[::-1]
    indexes = target.index

    res = target.copy()
    if len(source) < len(target):
        res.update(pd.Series([index.right for index in price_intervals], index=indexes[:len(source)]))
        res.iloc[len(source):] = 0
    else:
        res.update(pd.Series([index.right for index in price_intervals][:len(target)], index=indexes))

    return res


def get_tick_vol_table(target_table, df, lower_limit, upper_limit=None):
    """
        Возвращает список цен тиков выбросов для целевой таблицы. В первой ячейке цена самая поздняя по времени
        lower_limit - граничное значение больших объемов в тике
        upper_limit - граничное значение супер-больших объемов в тике
        target_table - целевая таблица
    """

    if upper_limit is None:
        condition = df['vol'] >= lower_limit
    else:
        condition = (df['vol'] >= lower_limit) & (df['vol'] < upper_limit)

    new_tick_vol_table = df.loc[condition][-len(target_table):]['close'][::-1]

    if len(new_tick_vol_table) == len(target_table):
        """
        предполагается, что целевая таблица уже была один раз заполнена. И, если
        в сдвинутом датафрейме меньше значимых объемов, мы не будем терять старые значения, вернём то,
        что было. Если значений столько же, то возвращаем новые значения (даже если совпадают).

        Больше, чем надо не может быть, т.к. мы обрезали лишнее выше.
        """
        return new_tick_vol_table

    else:
        return target_table


def split_on_frames(data: pd.DataFrame, path, minute_time_interval='1Min'):
    """
    для имитации реальной подачи данных и обновления таблицы признаков разбиваем
    уже полученные данные на минутные интервалы.
    data - все новые данные обработанные ф-цией prepare_data
    minute_time_interval - на какие таймфреймы разбиваем (в минутах)

    сохраняет каждый фрейм (минуту) в отдельный файл
    """
    # на случай чтения из файла
    data['DATETIME'] = pd.to_datetime(data['DATETIME'], format='%Y%m%d %H%M%S.%f')
    data['vol'] = data['vol'].astype(int)
    data['close'] = data['close'].astype(float)
    data['datetime_rounded'] = data.loc[:, 'DATETIME'].dt.floor(minute_time_interval)

    grouped = data.groupby('datetime_rounded').groups
    for name, group in grouped.items():
        print(name)
        data.loc[group].drop(columns='datetime_rounded').to_csv(f'{path}/{name.strftime("%y%m%d_%H_%M")}.csv')


def get_updated_week_df(old_data, data):
    """
    Добавляет данные за один фрейм (1 мин) и удаляем соответственно устаревший фрейм
    """

    # добавляем к прошлой неделе еще данные за новый период (1 мин)
    joined_df = pd.concat([old_data, data])

    # определяем время, до которого нужно сократить объединенный df (в данном случае на минуту)
    threshold_datetime = joined_df['DATETIME'].dt.floor('Min').unique()[1]

    #  отсекаем устаревшие значения (больше недели)
    shifted_df = joined_df[joined_df['DATETIME'] >= threshold_datetime]

    return shifted_df


def get_OHLC(data: pd.DataFrame):
    """
    возвращает open, high, low, close из данных
    """
    return data['close'].iloc[0], data['close'].max(), data['close'].min(), data['close'].iloc[-1]


def get_difference(close_price: float, data: pd.Series):
    """
    иногда data заполнено неполностью. "Хвосты" бывают нулями. Тогда получается
    значение разности в этих местах очень большими. Буду их обнулять.
    """

    row_data: np.ndarray = close_price - data.values
    row_data[np.where(data == 0)[0]] = 0

    return row_data


def get_updated_feature_table(table: pd.DataFrame,
                              date_time: pd.DatetimeIndex,
                              index: int,
                              week_peaks: pd.Series,
                              sup_vol_data: pd.Series,
                              vol_h_data: pd.Series,
                              tick_sup_vol: pd.Series,
                              tick_vol: pd.Series,
                              o, h, l, c):
    open_price, high, low, close_plrice = o, h, l, c
    table.loc[index, 'close'] = close_plrice
    table.loc[index, 'datetime'] = date_time
    table.loc[index, 'C-L'] = close_plrice - low
    table.loc[index, 'H-C'] = high - close_plrice
    table.loc[index, 'C-O'] = close_plrice - open_price
    table.loc[index, 'C-w_max_v_price'] = close_plrice - week_peaks.iloc[0]
    table.loc[index, 'C-sup_vol_pr_1':'C-sup_vol_pr_20'] = get_difference(close_plrice, sup_vol_data)
    table.loc[index, 'C-vol_pr_1':'C-vol_pr_20'] = get_difference(close_plrice, vol_h_data)
    table.loc[index, 'C-tick_sup_vol_pr_1':'C-tick_sup_vol_pr_8'] = get_difference(close_plrice, tick_sup_vol)
    table.loc[index, 'C-tick_vol_pr_1':'C-tick_vol_pr_20'] = get_difference(close_plrice, tick_vol)
    return table
