from flask import Flask, render_template, url_for, request, redirect, flash
from Classes_for_strategy import LadderStrategy, DatabaseHandler, StrategyController
from pybit.unified_trading import HTTP
from threading import Thread
import atexit
import os
import pandas as pd
from datetime import datetime, timedelta
import time
import requests
from collections import defaultdict


def get_some_top_bybit_symbols(category, limit):
    # Эндпоинт для получения данных о деривативах (V5 API)
    url = "https://api.bybit.com/v5/market/tickers"
    params = {"category": category, "limit": limit}

    try:
        response = requests.get(url, params=params)
        data1 = response.json()
        tickers = data1["result"]["list"]

        # Фильтруем и сортируем по объему (в descending порядке)
        sorted_tickers = sorted(tickers, key=lambda x: float(x["turnover24h"]), reverse=True)

        # Берем топ-100 символов
        top_symbols = [ticker["symbol"] for ticker in sorted_tickers[:params["limit"]]]
        return top_symbols

    except Exception as e:
        flash(f'Ошибка: {str(e)}', 'error')
        return []


def get_bybit_klines(symbol, interval, start_time, end_time, category="linear"):
    session = HTTP()
    data = []

    start_dt = datetime.strptime(start_time, '%Y-%m-%d')
    end_dt = datetime.strptime(end_time, '%Y-%m-%d')
    current_time = start_dt

    max_candles = 1000  # Максимум свечей за один запрос (лимит Bybit)

    while current_time < end_dt:
        next_time = current_time + max_candles * interval_to_timedelta(interval)

        # Корректируем next_time, если он превышает end_dt
        if next_time > end_dt:
            next_time = end_dt

        # Запрос данных
        resp = session.get_kline(
            category=category,
            symbol=symbol,
            interval=interval,
            start=int(current_time.timestamp() * 1000),
            end=int(next_time.timestamp() * 1000),
            limit=max_candles
        )

        if resp["retCode"] != 0:
            raise Exception(f"Ошибка: {resp['retMsg']}")

        # Добавляем данные только если они есть
        if resp["result"]["list"]:
            data.extend(resp["result"]["list"])

        # Обновляем current_time для следующего запроса
        current_time = next_time  # Исправлено: убрано добавление интервала
        time.sleep(0.1)

    # Обработка пустых данных
    if not data:
        return pd.DataFrame()

    # Создаем DataFrame
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    return df_sorted


def interval_to_timedelta(interval):
    intervals = {
        "1": 1, "3": 3, "5": 5, "15": 15, "30": 30,
        "60": 60, "120": 120, "240": 240, "D": 1440, "W": 10080
    }
    return timedelta(minutes=intervals[interval])


# Выносим функцию расчета ликвидации в глобальную область
def count_liquidation(percent, levels, leverage, after_step_persent_more, persent_more_by):
    count = round(100 / percent) + 1
    persent_down_v = [0] * count
    result = [1000] * count
    middle_cost = [100 - percent * i / 2 if i < levels else 0 for i in range(count)]
    middle_cost_down = [percent * i / 2 if i < levels else 1000 for i in range(count)]
    sum_persent = 0

    for i in range(count - 1):
        if i < levels:
            middle_cost_down[i] = sum_persent / 2
            middle_cost[i] = 100 - sum_persent / 2

            if i + 1 < after_step_persent_more:
                sum_persent += percent
            else:
                additional_percent = persent_more_by * (i - after_step_persent_more + 1)
                sum_persent += (percent + additional_percent + persent_more_by)

            persent_down_v[i] = levels / leverage / (i + 1) * 100
        else:
            persent_down_v[i] = persent_down_v[i - 1] if i > 0 else 0
        result[i] = middle_cost_down[i] + persent_down_v[i] * middle_cost[i] / 100

    return round(min(result), 2)


app = Flask(__name__)
app.secret_key = 'd3b07384d113edec49eaa6238ad5ff00'

# Инициализация DatabaseHandler как синглтона
db_handler = DatabaseHandler()

# Инициализируем контроллер стратегий
controller = StrategyController()


# Функция для запуска в фоновом потоке
def run_controller():
    while True:
        try:
            controller.run()
            time.sleep(60)
        except Exception as e:
            flash(f'Ошибка: {str(e)}', 'error')
            time.sleep(10)  # Пауза перед повторной попыткой


# Запускаем фоновый поток только один раз
if not app.debug or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
    controller_thread = Thread(target=run_controller, daemon=True)
    controller_thread.start()


# Очистка при завершении
@atexit.register
def cleanup():
    db_handler.close()
    print("Приложение завершает работу, соединения закрыты")


@app.route('/')
@app.route('/all_strategies')
def index():
    try:
        strategies = db_handler.load_active_strategies()
        current_route = request.path
        return render_template("index.html", strategies=strategies, current_route=current_route)
    except Exception as e:
        flash(f"Ошибка при загрузке стратегий: {str(e)}", "error")
        return render_template("index.html", strategies=[], current_route=current_route)


@app.route('/create_strategy', methods=['POST', 'GET'])
def create_strategy():
    if request.method == "POST":
        try:
            form_data = request.form

            # Проверка наличия обязательных полей
            required_fields = [
                'symbol', 'start_price', 'total_invest', 'steps',
                'percent_up', 'percent_down', 'leverage',
                'after_step_persent_more', 'persent_more_by'
            ]
            for field in required_fields:
                if field not in form_data:
                    raise ValueError(f"Не заполнено поле: {field}")

            # Валидация параметров
            symbol = form_data['symbol'].upper()
            start_price = float(form_data['start_price'])
            total_invest = float(form_data['total_invest'])
            steps = int(form_data['steps'])
            percent_up = float(form_data['percent_up'])
            percent_down = float(form_data['percent_down'])
            leverage = int(form_data['leverage'])
            after_step_persent_more = int(form_data['after_step_persent_more'])
            persent_more_by = float(form_data['persent_more_by'])

            # Создаем стратегию
            strategy = LadderStrategy(
                symbol=symbol,
                start_price=start_price,
                total_invest=total_invest,
                steps=steps,
                percent_down=percent_down,
                percent_up=percent_up,
                leverage=leverage,
                after_step_persent_more=after_step_persent_more,
                persent_more_by=persent_more_by
            )

            # Добавляем через контроллер с проверкой конфликтов
            controller.add_strategy(strategy)

            flash(f'Стратегия {symbol} успешно создана', 'success')
            return redirect('/all_strategies')

        except Exception as e:
            flash(f'Ошибка при создании стратегии: {str(e)}', 'error')
            current_route = request.path
            return render_template("create_strategy.html", form_data=request.form, current_route=current_route)

    current_route = request.path
    return render_template("create_strategy.html", current_route=current_route)


@app.route('/deactivate_strategy/<int:strategy_id>', methods=['POST'])
def deactivate_strategy(strategy_id):
    try:
        with controller.lock:  # Захватываем блокировку
            strategy = next((s for s in controller.strategies if s.id == strategy_id), None)
            if strategy:
                # Деактивируем и удаляем из списка
                strategy.deactivate()
                controller.db_handler.deactivate_strategy(strategy_id)
                controller.strategies.remove(strategy)
                print(f"Стратегия {strategy_id} удалена из контроллера")  # Логирование
            else:
                flash('Стратегия не найдена', 'error')
    except Exception as e:
        flash(f'Ошибка при деактивации: {str(e)}', 'error')
    return redirect('/all_strategies')


@app.route('/strategy_edit/<int:strategy_id>', methods=['GET', 'POST'])
def strategy_edit(strategy_id):
    try:
        # Получаем оригинальную стратегию
        original = next((s for s in controller.strategies if s.id == strategy_id), None)
        if not original:
            flash('Стратегия не найдена', 'error')
            return redirect('/all_strategies')

        if request.method == 'POST':
            try:
                # 1. Деактивируем оригинальную стратегию
                original.deactivate()
                controller.db_handler.deactivate_strategy(original.id)
                controller.strategies.remove(original)

                # 2. Создаем новую стратегию с обновленными параметрами
                new_strategy = LadderStrategy(
                    symbol=request.form['symbol'].upper(),
                    start_price=float(request.form['start_price']),
                    total_invest=float(request.form['total_invest']),
                    steps=int(request.form['steps']),
                    percent_down=float(request.form['percent_down']),
                    percent_up=float(request.form['percent_up']),
                    leverage=float(request.form['leverage']),
                    after_step_persent_more=int(request.form['after_step_persent_more']),
                    persent_more_by=float(request.form['persent_more_by'])
                )

                # 3. Добавляем новую стратегию
                controller.add_strategy(new_strategy)

                flash('Стратегия успешно обновлена', 'success')
                return redirect('/all_strategies')

            except Exception as e:
                flash(f'Ошибка при обновлении: {str(e)}', 'error')
                current_route = request.path
                return render_template("strategy_edit.html", strategy=original.to_dict(), current_route=current_route)

        # Для GET-запроса показываем текущие параметры
        current_route = request.path
        return render_template("strategy_edit.html", strategy=original.to_dict(), current_route=current_route)

    except Exception as e:
        flash(f'Ошибка: {str(e)}', 'error')
        return redirect('/all_strategies')


@app.route('/liquidation_count', methods=['GET', 'POST'])
def liquidation_count():
    result = None
    error = None
    form_data = {}

    if request.method == 'POST':
        try:
            form_data = {
                'percent_down': request.form.get('percent_down'),
                'levels_down': request.form.get('levels_down'),
                'leverage_down': request.form.get('leverage_down'),
                'after_step_persent_more': request.form.get('after_step_persent_more'),
                'persent_more_by': request.form.get('persent_more_by')
            }

            # Получаем данные из формы
            percent_down = float(request.form.get('percent_down'))
            levels_down = int(request.form.get('levels_down'))
            leverage_down = float(request.form.get('leverage_down'))
            after_step_persent_more = float(request.form.get('after_step_persent_more'))
            persent_more_by = float(request.form.get('persent_more_by'))

            # Функция расчета
            result = count_liquidation(percent_down, levels_down, leverage_down, after_step_persent_more,
                                       persent_more_by)
        except ValueError:
            flash("Пожалуйста, введите корректные числовые значения", "error")
        except ZeroDivisionError:
            flash("Процент не может быть нулевым", "error")
        except Exception as e:
            flash(f"Произошла ошибка: {str(e)}", "error")

    current_route = request.path
    return render_template("liquidation_count.html",
                           result=result,
                           error=error,
                           form_data=form_data,  # Передаем данные формы в шаблон
                           current_route=current_route)


@app.route('/stats', methods=['GET', 'POST'])
def stats():
    form_data = {}
    current_route = request.path
    coin_stats = []  # Для хранения статистики по монетам
    if request.method == 'POST':
        try:
            form_data = {
                'start_date': request.form.get('start_date'),
                'end_date': request.form.get('end_date')
            }
            # Преобразуем даты в datetime
            start_date = datetime.strptime(form_data['start_date'], '%Y-%m-%d')
            end_date = datetime.strptime(form_data['end_date'], '%Y-%m-%d') + timedelta(days=1)
            # Словарь для агрегации данных по монетам
            stats_dict = defaultdict(lambda: {'count': 0, 'total_profit': 0.0})
            try:
                # Читаем данные из файла
                with open('trades.log', 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) < 7:
                            continue
                        # Парсим дату сделки
                        trade_time = datetime.fromisoformat(parts[0])
                        # Проверяем попадание в период
                        if start_date <= trade_time <= end_date:
                            symbol = parts[5]
                            profit = float(parts[6])
                            # Учитываем только SELL-сделки (прибыль > 0)
                            if profit > 0:
                                stats_dict[symbol]['count'] += 1
                                stats_dict[symbol]['total_profit'] += profit
                # Форматируем данные для шаблона
                coin_stats = [
                    {
                        'symbol': symbol,
                        'count': data['count'],
                        'total_profit': data['total_profit']
                    }
                    for symbol, data in stats_dict.items()
                ]
                # Сортируем по символу
                coin_stats.sort(key=lambda x: x['symbol'])
            except FileNotFoundError:
                flash('Файл trades.log не найден', 'error')
            except Exception as e:
                flash(f'Ошибка обработки файла: {str(e)}', 'error')
        except ValueError:
            flash('Некорректный формат даты', 'error')
    return render_template('stats.html',
                           coin_stats=coin_stats,
                           form_data=form_data,
                           current_route=current_route)
@app.route('/tables', methods=['GET', 'POST'])
def tables():
    result = None
    error = None
    form_data = {}  # Словарь для данных формы
    show_results = False
    table_data = []

    def count_persent(df, n):
        if df.empty:
            return 0
        count = 0
        # Конвертируем все цены в float один раз
        highs = df['high'].astype(float).values
        lows = df['low'].astype(float).values

        minim = highs[0]
        maxim = lows[0]
        time_minim = 0
        time_maxim = 0

        for i in range(0, len(highs)):
            if highs[i] > maxim:
                maxim = highs[i]
                time_maxim = int(df['timestamp'][i].timestamp() * 1000)
            if lows[i] < minim:
                minim = lows[i]
                time_minim = int(df['timestamp'][i].timestamp() * 1000)
            if (maxim - minim) / minim * 100 > n:
                count += 1
                if time_maxim > time_minim:
                    minim = maxim
                    time_minim = time_maxim
                else:
                    maxim = minim
                    time_minim = time_maxim

        return count

    def get_persent_for_many_symbols(interval, start_time, end_time, category, persents, data):
        for j in persents:
            data[f'{j}%'] = []

        for i in range(len(data['symbols'])):
            df = get_bybit_klines(
                symbol=data['symbols'][i],
                interval=interval,
                start_time=start_time,
                end_time=end_time,
                category=category
            )
            for j in persents:
                data[f'{j}%'].append(count_persent(df, j))

    def create_data_frame(interval, start_time, end_time, category, n, limit):
        # Формируем список процентов, кратных 3, до n включительно
        persents = list(range(3, n + 1, 3))

        data = {'symbols': get_some_top_bybit_symbols(category, limit)}

        # Убедимся, что в data нет лишних ключей
        for key in list(data.keys()):
            if key != 'symbols':
                del data[key]

        get_persent_for_many_symbols(
            interval,
            start_time,
            end_time,
            category,
            persents,
            data
        )

        # Создаем DataFrame только с нужными столбцами
        df = pd.DataFrame(data)
        # Фильтруем столбцы: symbols и проценты из persents
        columns = ['symbols'] + [f'{p}%' for p in persents]
        df = df[columns]

        return df

    if request.method == 'POST':
        try:
            # Сохраняем данные формы
            form_data = {
                'start_date': request.form.get('start_date'),
                'end_date': request.form.get('end_date'),
                'coin_limit': request.form.get('coin_limit', '10'),
                'max_percent': request.form.get('max_percent', '15')
            }

            # Обработка данных формы
            start_date = form_data['start_date']
            end_date = form_data['end_date']
            coin_limit = int(form_data['coin_limit'])
            max_percent = int(form_data['max_percent'])

            # Генерируем актуальный список процентов
            persents = list(range(3, max_percent + 1, 3))

            data = create_data_frame(
                interval="60",
                start_time=start_date,
                end_time=end_date,
                category="linear",
                n=max_percent,
                limit=coin_limit
            )

            table_data = data.to_dict('records')
            show_results = True

        except Exception as e:
            error = f"Произошла ошибка: {str(e)}"

        current_route = request.path
        return render_template('tables.html',
                           table_data=table_data,
                           max_percent=max_percent if 'max_percent' in form_data else 15,
                           persents=persents,
                           show_results=show_results,
                           form_data=form_data,  # Передаем данные формы
                           error=error,
                           current_route=current_route)

    # GET-запрос: передаем начальные значения
    default_max_percent = 15
    default_persents = list(range(3, default_max_percent + 1, 3))

    current_route = request.path
    return render_template('tables.html',
                           show_results=False,
                           max_percent=default_max_percent,
                           persents=default_persents,
                           current_route=current_route)  # Значение по умолчанию


@app.route('/tables_with_steps', methods=['GET', 'POST'])
def tables_with_steps():
    def counted(df, start_price, total_invest, steps, percent_down, percent_up, leverage,
                after_step_persent_more, persent_more_by):

        if df.empty:
            return 0, 0, [], [], [], []

        count = 0
        highs = df['high'].astype(float).values
        lows = df['low'].astype(float).values
        levels_buy = [0 for _ in range(steps)]
        levels_sell = [0 for _ in range(steps)]
        invested = [False for _ in range(steps)]
        sum_persent = 0
        total_invest_sum = total_invest
        buy_events = []  # Список для хранения событий покупки
        sell_events = []  # Список для хранения событий продажи (timestamp, цена)
        buy_levels = []  # Список уровней покупки
        sell_levels = []  # Список уровней продажи

        for i in range(steps):
            levels_buy[i] = start_price * (1 - sum_persent / 100)

            if i + 1 < after_step_persent_more:
                sum_persent += percent_down
                levels_sell[i] = levels_buy[i] * (1 + percent_up / 100)
            else:
                # Последующие шаги: увеличиваем процент
                additional_percent = persent_more_by * (i - after_step_persent_more + 1)
                sum_persent += (percent_down + additional_percent + persent_more_by)
                levels_sell[i] = levels_buy[i] * (1 + (percent_up + additional_percent) / 100)

            buy_levels.append(levels_buy[i])
            sell_levels.append(levels_sell[i])

        for i in range(len(highs)):
            for j in range(len(levels_buy)):
                if lows[i] <= levels_buy[j] and not invested[j]:
                    invested[j] = True
                    buy_events.append((
                        df['timestamp'].iloc[i].strftime('%Y-%m-%d %H:%M:%S'),
                        float(levels_buy[j])  # Явное преобразование в float
                    ))
                if highs[i] >= levels_sell[j] and invested[j]:
                    count += 1
                    invested[j] = False

                    total_invest_step = total_invest_sum / steps
                    qty = (total_invest_step * leverage) / levels_buy[j]
                    profit = (levels_sell[j] - levels_buy[j]) * qty

                    total_invest_sum += profit
                    # Исправлено: преобразование времени в строку сразу
                    sell_events.append((
                        df['timestamp'].iloc[i].strftime('%Y-%m-%d %H:%M:%S'),
                        float(levels_sell[j])  # Явное преобразование в float
                    ))

        profit_percent = (total_invest_sum - total_invest) / total_invest * 100
        # Рассчитаем уровень ликвидации
        liquidation_percent = count_liquidation(
            percent_down, steps, leverage, after_step_persent_more, persent_more_by
        )
        liquidation_price = start_price * (1 - liquidation_percent / 100)

        return count, round(profit_percent, 2), buy_levels, sell_levels, buy_events, sell_events, liquidation_price

    current_route = request.path
    form_data = {}
    if request.method == 'POST':
        try:
            form_data = {
                'start_date': request.form.get('start_date', ''),
                'end_date': request.form.get('end_date', ''),
                'symbol': request.form.get('symbol', ''),
                'start_price': request.form.get('start_price', ''),
                'total_invest': request.form.get('total_invest', ''),
                'steps': request.form.get('steps', ''),
                'percent_down': request.form.get('percent_down', ''),
                'percent_up': request.form.get('percent_up', ''),
                'leverage': request.form.get('leverage', ''),
                'after_step_persent_more': request.form.get('after_step_persent_more', ''),
                'persent_more_by': request.form.get('persent_more_by', '')
            }

            start_date = request.form['start_date']
            end_date = request.form['end_date']
            symbol = request.form['symbol'].upper()
            start_price = float(request.form['start_price'])
            total_invest = float(request.form['total_invest'])
            steps = int(request.form['steps'])
            percent_down = float(request.form['percent_down'])
            percent_up = float(request.form['percent_up'])
            leverage = float(request.form['leverage'])
            after_step_persent_more = int(request.form['after_step_persent_more'])
            persent_more_by = float(request.form['persent_more_by'])

            df = get_bybit_klines(
                symbol=symbol,
                interval="60",
                start_time=start_date,
                end_time=end_date,
                category="linear"
            )

            if df.empty:
                return render_template('tables_with_steps.html',
                                       show_results=False,
                                       error_message="No data found for the selected period",
                                       current_route=current_route)

            count, profit, buy_levels, sell_levels, buy_events, sell_events, liquidation_price = counted(
                df, start_price, total_invest, steps, percent_down,
                percent_up, leverage, after_step_persent_more, persent_more_by
            )

            # Преобразование данных для графика
            price_data = {
                'time': df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'high': df['high'].astype(float).tolist(),
                'low': df['low'].astype(float).tolist()
            }

            # Явное преобразование всех чисел в float
            buy_levels = [float(x) for x in buy_levels]
            sell_levels = [float(x) for x in sell_levels]

            # Преобразование точек продажи - уже сделано в counted()
            sell_points = {
                'time': [e[0] for e in sell_events],
                'price': [float(e[1]) for e in sell_events]
            }

            # Создание JSON-сериализуемых данных
            chart_data = {
                'price_data': price_data,
                'buy_levels': buy_levels,
                'sell_levels': sell_levels,
                'buy_points': {
                    'time': [e[0] for e in buy_events],
                    'price': [float(e[1]) for e in buy_events]
                },
                'sell_points': {
                    'time': [e[0] for e in sell_events],
                    'price': [float(e[1]) for e in sell_events]
                },
                'liquidation_price': float(liquidation_price)
            }

            profit_invest = total_invest * profit / 100  # Рассчитываем прибыль в долларах

            return render_template('tables_with_steps.html',
                                   count=count,
                                   profit=profit,
                                   profit_invest=round(profit_invest, 2),
                                   chart_data=chart_data,
                                   show_results=True,
                                   form_data=form_data,
                                   current_route=current_route)

        except Exception as e:
            import traceback
            traceback.print_exc()
            flash(f"Произошла ошибка: {str(e)}", "danger")
            # При ошибке также передаем данные формы
            return render_template('tables_with_steps.html',
                                   show_results=False,
                                   form_data=form_data,
                                   current_route=current_route)

    return render_template('tables_with_steps.html',
                           show_results=False,
                           form_data=form_data,
                           current_route=current_route)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)