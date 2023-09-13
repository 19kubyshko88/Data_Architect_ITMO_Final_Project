import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def prepare_data(path_to_folder, files_list):
    """
    Чтение данных из файлов.
    :param path_to_folder:  - путь к папке с файлами
    :param files_list: - список файлов.
    :return:
    """
    col_names = ['ticker', 'per', 'date', 'time', 'open', 'high', 'low', 'close', 'vol']
    df = pd.concat([pd.read_table(f'{path_to_folder}/{f}',
                                  header=0, sep=',',
                                  names=col_names
                                  ) for f in files_list
                    ])

    df = df.drop(columns=['ticker', 'per', 'open', 'high', 'low'])

    if df.isna().any().any():
        print("Обнаружены строки с пустыпи значениями")
        print(df[df.isna().any(axis=1)])
        print("Удаление")
        df = df.drop(df[df.isna().any(axis=1)].index)
        print("Пустых значений:")
        print(df.isna().sum())

    df['date'] = df['date'].astype(str)
    df['time'] = df['time'].astype(str)
    df['vol'] = df['vol'].astype(int)
    # Добавляем колонку DATETIME
    df['DATETIME'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y%m%d %H%M%S.%f')
    # Убираем лишние столбцы
    df = df.drop(columns=['date', 'time'])
    mask1 = (df['DATETIME'].dt.time < pd.to_datetime('10:00').time())
    mask2 = ((df['DATETIME'].dt.time > pd.to_datetime('18:40').time()) &
             (df['DATETIME'].dt.time < pd.to_datetime('19:05').time()))

    mask = ~(mask1 | mask2)
    df = df[mask]

    return df


def get_horizontal_volumes(df, price_step: int | float = 1, timeframe: str = '1h') -> tuple:
    """
    Функция генерирует данные для построения баров распределения объемов сделок

    :Parameters:
        start_price - нижняя цена, от которой будут расчитываться интервалы
        price_step - с каким шагом по цене группировать объемы.
        timeframe - таймфрейм графика. Например '1s'/'1min'/'1h'


    :return:
        all_vols: MultiIndex Series names=['DATETIME', 'close'] - просуммированные объемы по фрейму и
        по интервалу цены (для построения свечей)


        TODO
        Весь период нужно будет ограничить (годом для часовика, неделей для минутного таймфрейма).
        Считаем, что за год ситуация значительно меняется и более ранние данные становятся менее значимыми.
    """

    # нижняя цена, от которой будут расчитываться интервалы
    # подбираем исходя из графика за неделю и минимального шага цены бумаги.
    start_price = np.floor(df['close'].min())  # - np.floor(df['close'].min())%10

    all_vols = df.groupby([df['DATETIME'].dt.floor(timeframe),
                           pd.cut(df['close'], bins=np.arange(start_price, df['close'].max(), price_step))])[
        'vol'].sum()

    return all_vols


def get_anomalies(data: pd.Series, threshold: float = 3):
    """
    Тест Граббса для поиска аномалий в данных.
    Если предполагается нормальное распределение, то ищем выбросы за пределами 3-сигма.
    :param data: - исходный столбец данных
    :return: pd.Series - выбросы за 3-сигма
    """
    mean = np.mean(data)
    std = np.std(data)

    lower_limit = mean - threshold * std
    upper_limit = mean + threshold * std

    # print("Lower limit:", lower_limit)
    # print("Upper limit:", upper_limit)

    anomalies = data[(data < lower_limit) | (data > upper_limit)].sort_values()

    # print("Detected anomalies:", anomalies)
    return anomalies


def sum_vol_by_price(h_vol: pd.Series):
    """
    Группировка объемов на уровнях
    MultiIndex:('DATETIME':Timestamp, close :Interval(closed='right'))
    Суммирует объемы по уровню цены. Нужно для выявления уровней, которые накопили объем не за один раз.
    Если объем превысит sup_quantile, то уровень перейдет в таблицу сверх-объемов.
    """
    grouped_h_vol = h_vol.groupby(level='close').sum()
    grouped_h_vol = grouped_h_vol.loc[
        grouped_h_vol.ne(0)]  # откуда-то берутся уровни с объемом 0, как в свече. Удаляем их.

    datetime_index = []
    for close_value in grouped_h_vol.index:
        datetime_index.append(h_vol.loc[:, close_value].index.sort_values()[-1])

    grouped_h_vol.index = pd.MultiIndex.from_arrays([datetime_index, grouped_h_vol.index], names=['DATETIME', 'close'])

    return grouped_h_vol


def get_horizontal_vol_anomal_candles(all_vols, quantile_coeff=0.95, super_quantile=0.99):
    """
        quntile_coeff - квантиль, определяющая выбросы. Определяется эмпирическим путём, глядя на график объемов.
                    Выбросы должны влиять на движение цены (уровни сопротивления/поддержки)
        super_quantile - исключительные выбросы

        quantile_coeff=0.75, super_quantile=0.9 для 1h
        quantile_coeff = 0.95, super_quantile = 0.99 для 1min
        :return:
                anomal_vol: MultiIndex Series - свечи с большими объемами.
                            Остальные объемы обнулены, но строки не удалены!
                sup_anomal_vol: MultiIndex Series - сверх большие объемы.
                                Остальные объемы обнулены, но строки не удалены!
                min_q, sup_q - граница больших и очень больших горизонтальных объемов, соответственно.

        TODO:
            попробовать не обнулять nan в свечах. Может быстрее отрисовывать будет.
    """
    # candle_form = pd.Series(np.zeros_like(all_vols), index=all_vols.index)

    anomalies = get_anomalies(all_vols.loc[all_vols.ne(0)].dropna())

    min_q = np.quantile(anomalies, quantile_coeff)
    anomal_vol_candle = all_vols.where(all_vols >= min_q)

    sup_q = np.quantile(anomalies, super_quantile)
    sup_anomal_vol_candle = anomal_vol_candle.where(anomal_vol_candle >= sup_q)

    anomal_vol_candle = anomal_vol_candle.fillna(0).astype(int)
    sup_anomal_vol_candle = sup_anomal_vol_candle.fillna(0).astype(int)

    return anomal_vol_candle, sup_anomal_vol_candle, min_q, sup_q


def get_tick_vol_anomalies_threshold(df, quantile_val=0.99, sup_quantile_val=0.995):
    """
    Функция выбирает самые значимые объемы в тиках из выбросов.
    :param df:
    :param quantile_val: - квантили, определяющие значимые выбросы
    :param sup_quantile_val:  - квантили, определяющие значимые супервыбросы
    :return:  пороговые значения объемов для определения значимых сделок
            anomal_tick_vol_threshold,
            sup_tick_anomal_vol_threshold
    """
    tick_vol_anomalies = get_anomalies(df['vol'])
    anomal_tick_vol_threshold = tick_vol_anomalies.quantile(quantile_val)
    sup_tick_anomal_vol_threshold = tick_vol_anomalies.quantile(sup_quantile_val)

    return anomal_tick_vol_threshold, sup_tick_anomal_vol_threshold


def get_vol_spred(all_vols):
    """
    Возвращает суммарный объем за весь период (для барчара общего распределения)
    (для минутного графика - неделя, для часовика - год)
    return:
        vols_spread: Series
    """
    # start_price = all_vols.index.get_level_values(1)[0].left
    # vols_spread = df.groupby(
    #                     pd.cut(df['close'], bins=np.arange(start_price, df['close'].max(), price_step)))['vol'].sum()

    vols_spread = pd.Series(data=np.zeros_like(all_vols.loc[all_vols.index.get_level_values(level=0)[0]]),
                            index=all_vols.index.get_level_values(level=1).unique())
    periods = all_vols.index.get_level_values(level=0).unique()
    for time in periods:
        vols_spread = vols_spread.add(all_vols.loc[time])

    return vols_spread


def plot_volume_profile(df: pd.DataFrame,
                        all_vols: pd.Series,
                        timeframe: str,
                        anomal_vol: pd.Series,
                        sup_anomal_vol: pd.Series,
                        vols_spread,
                        plot_values=False,
                        **kwargs):
    """
    Функция для отрисовки свечей с горизонтальными объемами - volume profile
    Таймфреймы меньше 1h печатает очень медленно!

    df - тиковый датафрейм для поиска open-close
    all_vols - сгруппированные по цене и просуммированные объемы  MultiIndex:('DATETIME':Timestamp,
                                                                                close :Interval(closed='right'))
    anomal_vol - выбросы горизонтальных объемов MultiIndex:('DATETIME':Timestamp,
                                                                                close :Interval(closed='right'))
    sup_anomal_vol - большие выбросы горизонтальных объемов MultiIndex:('DATETIME':Timestamp,
                                                                                close :Interval(closed='right'))
    vols_spread- Series распределения объемов по ценам за весь df Индексы - интервалы цен.
    plot_values - отображать или нет надписи на барах.

    kwargs:
     fontsize: 10 - normal
     figsize :tuple(width, height) (60, 10)
     line_width
    """

    fontsize = kwargs.get('fontsize', 10)
    figsize = kwargs.get('figsize', (60, 10))
    line_width = kwargs.get('line_width', 10)
    # Д.т.ч. объемы отрисовывались пропорционально, установим по Х этот максимум
    max_width = all_vols.max()

    # Две системы координат: для объемов и open/close
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=figsize, sharex='all')
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.rc('ytick', labelsize=fontsize * 2)
    # Свечи берем по времени. Получаем их обозначения.
    bar_time_indexes = all_vols.index.get_level_values(0).unique()

    bar_count = len(bar_time_indexes)  # количество свечей
    bar_count += 1  # для суммирующего распределения.

    i = 1  # счетчик свечей

    for time in bar_time_indexes:

        vols = all_vols[time]
        plt.subplot(1, bar_count, i, xlim=(-1, max_width))  #

        ax = vols.plot(kind='barh',
                       color='b',
                       width=1,
                       zorder=1,
                       fontsize=fontsize
                       )

        big_vol_idx = anomal_vol[time]
        if big_vol_idx.sum() > 0:  # если есть выбросы по объему, то отрисовать их
            ax = big_vol_idx.plot(kind='barh',
                                  color='orange',
                                  width=1,
                                  sharey=True,
                                  zorder=2,
                                  fontsize=fontsize
                                  )

        sup_vol_idx = sup_anomal_vol[time]
        if sup_vol_idx.sum() > 0:  # если есть супер-выбросы по объему, то отрисовать их
            ax = sup_vol_idx.plot(kind='barh',
                                  color='r',
                                  width=1,
                                  sharey=True,
                                  zorder=3
                                  )

        # # отрисовка open-close

        ax2 = ax.twiny()
        ax2 = ax.twinx()
        # ax3.set_xlim(-1, max_width)
        # ax3.set_xlim(xmin=-vols_spread.max(), xmax=0)

        ax2.set_yticks([i.mid for i in all_vols[time].index], all_vols[time].index)
        ax2.set_ylim(ymin=all_vols[time].index[0].left, ymax=all_vols[time].index[-1].right)
        # high = vols[vols.ne(0)].first_valid_index().mid
        # low = vols[vols.ne(0)].last_valid_index().mid
        open_price = df[df['DATETIME'].dt.floor(timeframe) == time].iloc[0]['close']
        close_price = df[df['DATETIME'].dt.floor(timeframe) == time].iloc[-1]['close']

        color = 'green' if close_price > open_price else 'red'

        ax2.plot([-1, -1], [open_price, close_price],
                 color=color,
                 linewidth=line_width,

                 )
        ax2.yaxis.set_visible(False)
        ax2.set_frame_on(False)

        ax.set_xlabel(f"{time.strftime('%m/%d')}",
                      fontsize=fontsize, rotation=90)  # подписываем день бара
        ax.set_title(f"{time.strftime('%H:%M')}",  # подписываем время бара
                     fontsize=fontsize)
        # Подписи к объемам.
        if plot_values:
            for bar_i, v in enumerate(vols):
                if v > 0:
                    ax.text(0, bar_i, f"{v}", ha='left', va='center', color='y', fontweight='semibold',
                            stretch=1,
                            fontsize=fontsize)
        if i > 1:  # не отображать рамку и оси на последующих свечах
            ax.yaxis.set_visible(False)
            ax.set_frame_on(False)
            ax2.yaxis.set_visible(False)
            ax2.set_frame_on(False)

            ax.xaxis.set_ticklabels([])
            ax2.xaxis.set_ticklabels([])
        i += 1

    # Суммирующий барчар
    plt.subplot(1, bar_count, bar_count)

    ax = (vols_spread * (-1)).plot(kind='barh', color='g', width=1)
    ax.set_xlim(xmin=-vols_spread.max(), xmax=0)
    max_bar = vols_spread.where(vols_spread == vols_spread.max())
    ax = (-max_bar).plot(kind='barh', color='r', width=1)
    ax.set_title("Сумма объемов")
    ax.yaxis.set_ticks_position('right')
    ax.yaxis.set_label_position('right')
    plt.rcParams['ytick.direction'] = 'out'

    plt.subplots_adjust(wspace=0.05)

    plt.show();
