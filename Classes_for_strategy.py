from flask import flash
from pybit.unified_trading import HTTP
from datetime import datetime
from keys import api, secret
from threading import Lock
import sqlite3
import time
import json
import math


class DatabaseHandler:
    """Класс для работы с базой данных стратегий."""

    _instance = None
    _lock = Lock()

    def __new__(cls, db_name='strategies.db'):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DatabaseHandler, cls).__new__(cls)
                cls._instance.initialized = False
        return cls._instance

    def __init__(self, db_name='strategies.db'):
        if not self.initialized:
            self.db_name = db_name
            self.conn = None
            self._connect()
            self.create_tables()
            self.initialized = True

    def _connect(self):
        """Устанавливает соединение с базой данных."""
        try:
            self.conn = sqlite3.connect(self.db_name, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
        except Exception as e:
            print(f"Ошибка подключения к базе данных: {e}")
            raise

    def get_strategies_stats(self, start_date, end_date):
        """Возвращает статистику стратегий за период"""
        self._ensure_connection()
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT 
                    id,
                    symbol,
                    active,
                    created_at,
                    count AS total_count,
                    total_profit,
                    (SELECT COUNT(*) FROM strategy_events 
                     WHERE strategy_id = strategies.id 
                     AND event_time BETWEEN ? AND ?) AS period_count
                FROM strategies
                WHERE created_at <= ?
                ORDER BY created_at DESC
            ''', (start_date, end_date, end_date))

            return cursor.fetchall()
        except Exception as e:
            print(f"Ошибка получения статистики: {e}")
            return []

    def get_strategy_by_id(self, strategy_id):
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM strategies WHERE id = ?', (strategy_id,))
        row = cursor.fetchone()
        if row:
            buy_levels = json.loads(row['buy_levels'])
            return {
                'id': row['id'],
                'symbol': row['symbol'],
                'start_price': row['start_price'],
                'total_invest': row['total_invest'],
                'steps': row['steps'],
                'percent_down': row['percent_down'],
                'percent_up': row['percent_up'],
                'leverage': row['leverage'],
                'after_step_persent_more': row['after_step_persent_more'],
                'persent_more_by': row['persent_more_by'],
                'buy_levels': buy_levels,
                'active': row['active'],
                'count': row['count']
            }
        return None

    def increment_count(self, strategy_id):
        """Увеличивает счетчик и записывает событие"""
        self._ensure_connection()
        cursor = self.conn.cursor()
        try:
            # Обновляем счетчик
            cursor.execute('''
                UPDATE strategies 
                SET count = count + 1 
                WHERE id = ?
            ''', (strategy_id,))

            # Записываем событие
            cursor.execute('''
                INSERT INTO strategy_events (strategy_id, event_type)
                VALUES (?, 'sell')
            ''', (strategy_id,))

            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise

    def _ensure_connection(self):
        """Проверяет и восстанавливает соединение при необходимости."""
        try:
            if self.conn is None:
                self._connect()
            # Простая проверка соединения
            self.conn.execute("SELECT 1").fetchone()
        except (sqlite3.ProgrammingError, sqlite3.OperationalError):
            self._connect()

    def create_tables(self):
        """Создает таблицы, если они не существуют."""
        self._ensure_connection()
        cursor = self.conn.cursor()

        # Создаем таблицу strategies (если нужно)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                start_price REAL NOT NULL,
                total_invest REAL NOT NULL,
                steps INTEGER NOT NULL,
                percent_down REAL NOT NULL,
                percent_up REAL NOT NULL,
                leverage REAL NOT NULL,
                after_step_persent_more INTEGER NOT NULL,
                persent_more_by REAL NOT NULL,
                buy_levels TEXT NOT NULL,
                active TEXT NOT NULL,
                count INTEGER DEFAULT 0,
                total_profit REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Добавляем создание таблицы strategy_events
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                event_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(strategy_id) REFERENCES strategies(id)
            )
        ''')

        self.conn.commit()

    def load_active_strategies(self):
        """Загружает активные стратегии из базы данных."""
        self._ensure_connection()
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT id, symbol, start_price, total_invest, steps, percent_down, 
                       percent_up, leverage, after_step_persent_more, persent_more_by, buy_levels, count
                FROM strategies
                WHERE active = 'active'
            ''')
            rows = cursor.fetchall()
            strategies = []
            for row in rows:
                strategies.append({
                    'id': row['id'],
                    'symbol': row['symbol'],
                    'start_price': row['start_price'],
                    'total_invest': row['total_invest'],
                    'steps': row['steps'],
                    'percent_down': row['percent_down'],
                    'percent_up': row['percent_up'],
                    'leverage': row['leverage'],
                    'after_step_persent_more': row['after_step_persent_more'],
                    'persent_more_by': row['persent_more_by'],
                    'buy_levels': json.loads(row['buy_levels']),
                    'count': row['count']
                })
            return strategies
        except Exception as e:
            print(f"Ошибка при загрузке стратегий: {e}")
            return []

    def add_strategy(self, strategy_data):
        """Добавляет новую стратегию в базу данных."""
        cursor = self.conn.cursor()
        buy_levels_json = json.dumps(strategy_data['buy_levels'])
        cursor.execute('''
            INSERT INTO strategies (symbol, start_price, total_invest, steps, percent_down, percent_up, leverage, after_step_persent_more, persent_more_by, buy_levels, active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            strategy_data['symbol'],
            strategy_data['start_price'],
            strategy_data['total_invest'],
            strategy_data['steps'],
            strategy_data['percent_down'],
            strategy_data['percent_up'],
            strategy_data['leverage'],
            strategy_data['after_step_persent_more'],
            strategy_data['persent_more_by'],
            buy_levels_json,
            'active'
        ))
        self.conn.commit()
        return cursor.lastrowid

    def deactivate_strategy(self, strategy_id):
        """Помечает стратегию как неактивную."""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE strategies
            SET active = 'inactive'
            WHERE id = ?
        ''', (strategy_id,))
        self.conn.commit()

    def update_strategy(self, strategy_id, updated_data):
        cursor = self.conn.cursor()
        buy_levels = updated_data.get('buy_levels', [])
        buy_levels_json = json.dumps(buy_levels) if buy_levels else None
        total_invest = updated_data.get('total_invest')
        total_profit = updated_data.get('total_profit')

        try:
            cursor.execute('''
                UPDATE strategies
                SET 
                    symbol = COALESCE(?, symbol),
                    start_price = COALESCE(?, start_price),
                    total_invest = COALESCE(?, total_invest),
                    steps = COALESCE(?, steps),
                    percent_down = COALESCE(?, percent_down),
                    percent_up = COALESCE(?, percent_up),
                    leverage = COALESCE(?, leverage),
                    after_step_persent_more = COALESCE(?, after_step_persent_more),
                    persent_more_by = COALESCE(?, persent_more_by),
                    buy_levels = COALESCE(?, buy_levels),
                    total_profit = COALESCE(?, total_profit),
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (
                updated_data.get('symbol'),
                updated_data.get('start_price'),
                total_invest,
                updated_data.get('steps'),
                updated_data.get('percent_down'),
                updated_data.get('percent_up'),
                updated_data.get('leverage'),
                updated_data.get('after_step_persent_more'),
                updated_data.get('persent_more_by'),
                buy_levels_json,
                total_profit,
                strategy_id
            ))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Failed to update strategy: {str(e)}") from e

    def update_total_invest(self, strategy_id, new_total_invest):
        """Обновляет total_invest в базе данных"""
        self._ensure_connection()
        cursor = self.conn.cursor()
        try:
            cursor.execute('''
                UPDATE strategies
                SET total_invest = ?, 
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (new_total_invest, strategy_id))
            self.conn.commit()
        except Exception as e:
            print("Ошибка при обновлении total_invest: {e}")
            self.conn.rollback()
            raise

    def add_profit(self, strategy_id, profit):
        """Добавляет прибыль к общей прибыли стратегии"""
        self._ensure_connection()
        cursor = self.conn.cursor()
        try:
            cursor.execute('''
                UPDATE strategies
                SET total_profit = total_profit + ?
                WHERE id = ?
            ''', (profit, strategy_id))
            self.conn.commit()
        except Exception as e:
            print(f"Ошибка при обновлении прибыли: {e}")
            self.conn.rollback()
            raise

    def close(self):
        """Закрывает соединение с базой данных."""
        if self.conn:
            self.conn.close()
            self.conn = None


class LadderStrategy:
    def __init__(self,
                 symbol: str,
                 start_price: float,
                 total_invest: float,
                 steps: int,
                 percent_down: float,
                 percent_up: float,
                 leverage: float,
                 after_step_persent_more: int,
                 persent_more_by: float,
                 db_handler: DatabaseHandler = None,
                 from_db: bool = False):
        self.db_handler = db_handler  # Сохраняем ссылку на обработчик БД
        self.session = HTTP(testnet=False, api_key=api, api_secret=secret)
        self.symbol = symbol
        self.start_price = start_price
        self.total_invest = total_invest
        self.steps = steps
        self.leverage = leverage
        self.percent_down = percent_down
        self.percent_up = percent_up
        self.step_investment = self.total_invest / self.steps
        self.buy_levels = []
        self.active_positions = []
        self.current_price = 0
        self.active = 'active'
        self.id = None  # ID из базы данных
        self.after_step_persent_more = after_step_persent_more
        self.persent_more_by = persent_more_by

        # Инициализация уровней только для новых стратегий
        if not from_db:
            try:
                self._calculate_buy_levels()
                print(f"Новая стратегия создана для {symbol}")
            except Exception as e:
                print(f"Ошибка инициализации: {e}")
                raise

    def _calculate_buy_levels(self):
        """Рассчитывает уровни покупки."""
        if self.after_step_persent_more >= self.steps:
            raise ValueError("Шаг усиления должен быть меньше общего количества шагов")

        self.buy_levels = []  # Явно инициализируем как список
        price_precision = self.get_precisions()[0]

        sum_persent = 0
        for i in range(self.steps):

            price = self.start_price * (1 - sum_persent / 100)

            self.buy_levels.append({
                'level': i + 1,
                'target_price': round(price, price_precision),
                'invested': False,
                'buy_price': None,
                'sell_price': None,
                'qty': None
            })

            if i + 1 < self.after_step_persent_more:
                sum_persent += self.percent_down
            else:
                additional_percent = self.persent_more_by * (i - self.after_step_persent_more + 1)
                sum_persent += (self.percent_down + additional_percent + self.persent_more_by)

    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'start_price': self.start_price,
            'total_invest': self.total_invest,
            'steps': self.steps,
            'percent_down': self.percent_down,
            'percent_up': self.percent_up,
            'leverage': self.leverage,
            'after_step_persent_more': self.after_step_persent_more,
            'persent_more_by': self.persent_more_by,
            'buy_levels': self.buy_levels  # Добавляем уровни в вывод
        }

    @classmethod
    def from_dict(cls, data, db_handler=None):
        """Создает стратегию из данных базы."""
        strategy = cls(
            symbol=data['symbol'],
            start_price=data['start_price'],
            total_invest=data['total_invest'],
            steps=data['steps'],
            percent_down=data['percent_down'],
            percent_up=data['percent_up'],
            leverage=data['leverage'],
            after_step_persent_more=data['after_step_persent_more'],
            persent_more_by=data['persent_more_by'],
            db_handler=db_handler,  # Передаем обработчик БД
            from_db=True
        )
        strategy.id = data['id']
        strategy.buy_levels = data.get('buy_levels', [])
        strategy.count = data.get('count', 0)  # Загружаем значение счетчика
        strategy.total_profit = data.get('total_profit', 0)
        return strategy

    def deactivate(self):
        """Деактивирует стратегию: закрывает позиции и помечает как неактивную."""
        print(f"Деактивация стратегии для {self.symbol}...")

        # Закрываем открытые позиции
        for level in self.buy_levels:
            if level['invested']:
                self._place_sell_order(level)

        self.active = 'inactive'
        print(f"Стратегия для {self.symbol} деактивирована")

    # получение правильной цены и объёма (кол-во знаков после запятой)
    def get_precisions(self):
        try:
            resp = self.session.get_instruments_info(category='linear', symbol=self.symbol)
            instrument = resp['result']['list'][0]

            min_qty = float(instrument['lotSizeFilter']['minOrderQty'])
            qty_step = float(instrument['lotSizeFilter']['qtyStep'])
            min_order_value = 5.0  # Минимальная сумма ордера в USDT для Bybit

            def parse_precision(value_str):
                if '.' in value_str:
                    return len(value_str.split('.')[1].rstrip('0'))
                return 0

            return (
                parse_precision(instrument['priceFilter']['tickSize']),
                parse_precision(instrument['lotSizeFilter']['qtyStep']),
                min_qty,
                min_order_value
            )
        except Exception as e:
            print(f"Ошибка получения параметров инструмента {self.symbol}: {str(e)}", "error")
            return 4, 4, 0.001, 5.0  # Значения по умолчанию

    # получение актуальной стоимости актива, один раз за проход монеты
    def _get_current_price(self):
        try:
            ticker = self.session.get_tickers(category='linear', symbol=self.symbol)
            return float(ticker['result']['list'][0]['markPrice'])
        except Exception as e:
            print(f"Error getting price: {e}")
            return None

    # выставление ордера покупки по цене уровня (print)
    def _place_buy_order(self, level):
        try:
            price_precision, qty_precision, min_qty, min_order_value = self.get_precisions()
            qty = round((self.step_investment * self.leverage) / level['target_price'], qty_precision)
            order_value = qty * level['target_price']

            # Проверка минимального количества
            if qty < min_qty:
                print(f"Количество {qty} меньше минимального {min_qty} для {self.symbol}", "error")
                return False

            # Проверка минимальной суммы ордера (5 USDT)
            if order_value < min_order_value:
                required_qty = math.ceil(
                    (min_order_value / level['target_price']) * 10 ** qty_precision) / 10 ** qty_precision
                print(
                    f"Сумма ордера {order_value:.2f}USDT < минимальной {min_order_value}USDT для {self.symbol}. "
                    f"Увеличьте объем инвестиций или используйте минимум {required_qty} контрактов",
                    "error"
                )
                return False

            order = self.session.place_order(
                category='linear',
                symbol=self.symbol,
                side='Buy',
                orderType='Limit',
                qty=qty,
                price=round(level['target_price'], price_precision),
                timeInForce='GTC'
            )

            if 'result' not in order or 'orderId' not in order['result']:
                error_msg = order.get('retMsg', 'Неизвестная ошибка при размещении ордера')
                print(f"Ошибка размещения ордера на покупку ({self.symbol}): {error_msg}", "error")
                return False

            if level['level'] < self.after_step_persent_more:
                sell_price_level = round(level['target_price'] * (1 + self.percent_up / 100), price_precision)
            else:
                additional_percent = self.persent_more_by * (level['level'] - self.after_step_persent_more)
                sell_price_level = round(level['target_price'] * (1 + (self.percent_up + additional_percent) / 100),
                                         price_precision)

            level.update({
                'invested': True,
                'buy_price': level['target_price'],
                'sell_price': sell_price_level,
                'qty': qty
            })

            print(f"Placing BUY order {level['target_price']} for level {level['level']} "
                  f"at {level['target_price']} ({level['qty']} {self.symbol})")

            self.log_trade('BUY', level['level'], level['target_price'], qty, 0)
            return True


        except Exception as e:
            error_msg = f"Ошибка размещения ордера на покупку ({self.symbol}): {str(e)}"
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                try:
                    error_data = json.loads(e.response.text)
                    error_msg = error_data.get('retMsg', error_msg)
                except:
                    pass
            print(error_msg, "error")
            return False

    # выставление ордера продажи по цене уровня (print)
    def _place_sell_order(self, level):
        try:

            print(f"Placing SELL order {level['sell_price']} for level {level['level']} "
                  f"at {level['sell_price']} ({level['qty']} {self.symbol})")

            order = self.session.place_order(
                category='linear',
                symbol=self.symbol,
                side='Sell',
                orderType='Limit',
                qty=level['qty'],
                price=level['sell_price'],
                timeInForce='GTC'
            )

            if 'result' not in order or 'orderId' not in order['result']:
                print("Failed to place buy order")
                return False

                # Успешная продажа - увеличиваем счетчик и обновляем инвестиции
            if 'result' in order and 'orderId' in order['result']:
                # Рассчитываем прибыль
                profit = (level['sell_price'] - level['buy_price']) * level['qty']
                # Обновляем общий объем инвестиций
                self.total_invest += profit
                self.step_investment = self.total_invest / self.steps

                # Обновляем данные в базе
                if self.db_handler and self.id:
                    self.db_handler.increment_count(self.id)
                    self.db_handler.update_total_invest(self.id, self.total_invest)
                    self.db_handler.add_profit(self.id, profit)

            self.log_trade('SELL', level['level'], level['sell_price'], level['qty'], profit)

            level['invested'] = False
            level['buy_price'] = None
            level['sell_price'] = None
            level['qty'] = None

            return True

        except Exception as e:
            print(f"Error placing sell order: {e}")
            return False

    # реализация алгоритма на покупку/продажу
    def check_conditions(self):
        self.current_price = self._get_current_price()
        self.print_status()
        if not self.current_price:
            return

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {self.symbol} Current price: {self.current_price}")
        has_position = self._get_open_position()

        if has_position:
            for level in self.buy_levels:
                if level['invested'] and level['sell_price'] is not None:
                    if self.current_price > level['sell_price'] and self._place_sell_order(level):
                        print("Sell order placed successfully")
        else:
            print(f"No open position for {self.symbol}, skipping sell checks")

        for level in self.buy_levels:
            if not level['invested']:
                if self.current_price < level['target_price'] and self._place_buy_order(level):
                    print("Buy order placed successfully")

    # логирование в консоли
    def print_status(self):
        try:
            status = [
                "\n" + "=" * 30,
                f"Strategy status for {self.symbol}",
                f"Invested: {self.total_invest}",
                f"Current price: {self.current_price or 'N/A'}",
                f"Active levels: {sum(1 for l in self.buy_levels if l['invested'])}/{self.steps}",
                "=" * 30 + "\n"
            ]
            print('\n'.join(status))
        except Exception as e:
            print(f"Status error: {e}")

    # огирование в файл trades.log
    def log_trade(self, action, level, price, qty, profit):
        try:
            log_line = f"{datetime.now().isoformat()},{action},{level},{price},{qty},{self.symbol},{profit}\n"

            with open('trades.log', 'a', encoding='utf-8') as f:
                f.write(log_line)
        except Exception as e:
            print(f"Logging error: {e}")

    # проверка открытых позицый(нужна для выставления ордеров на продажу)
    def _get_open_position(self):
        try:
            response = self.session.get_positions(category='linear', symbol=self.symbol)
            if float(response['result']['list'][0]['size']) > 0:
                return True
            return False
        except Exception as e:
            print(f"Error checking position: {e}")
            return False


class StrategyController:
    def __init__(self):
        self.strategies = []
        self.lock = Lock()
        self.db_handler = DatabaseHandler()
        self._load_strategies_from_db()
        self.running = True

    def _load_strategies_from_db(self):
        """Загружает активные стратегии из базы данных."""
        strategies_data = self.db_handler.load_active_strategies()
        for data in strategies_data:
            strategy = LadderStrategy.from_dict(data, db_handler=self.db_handler)
            self._synchronize_strategy(strategy)
            self.strategies.append(strategy)
            print(f"Загружена стратегия из БД: {strategy.symbol}")

    def _synchronize_strategy(self, strategy):
        """Синхронизирует состояние стратегии с биржей."""
        has_position = strategy._get_open_position()
        if not has_position:
            # Сбрасываем уровни, если позиция закрыта
            for level in strategy.buy_levels:
                if level['invested']:
                    level.update({
                        'invested': False,
                        'buy_price': None,
                        'sell_price': None,
                        'qty': None
                    })
            # Обновляем в базе данных
            self.db_handler.update_strategy(strategy.id, {'buy_levels': strategy.buy_levels})  # <- Исправлено!

    def add_strategy(self, new_strategy):
        """Добавляет новую стратегию с проверкой конфликтов"""
        existing = [s for s in self.strategies if s.symbol == new_strategy.symbol]

        if existing:
            current_params = new_strategy.to_dict()
            # Удаляем временные поля
            for key in ['buy_levels', 'active', 'id']:
                current_params.pop(key, None)

            for old_strategy in existing:
                old_params = old_strategy.to_dict()
                # Удаляем временные поля
                for key in ['buy_levels', 'active', 'id']:
                    old_params.pop(key, None)

                if old_params == current_params:
                    print('Стратегия с такими параметрами уже существует', 'error')
                    return

                # Деактивируем старые стратегии
                old_strategy.deactivate()
                self.db_handler.deactivate_strategy(old_strategy.id)
                self.strategies.remove(old_strategy)

        # Добавляем новую стратегию
        strategy_data = new_strategy.to_dict()  # Используем обновленный to_dict()
        strategy_id = self.db_handler.add_strategy(strategy_data)
        new_strategy.id = strategy_id
        new_strategy.db_handler = self.db_handler  # Привязываем обработчик БД
        self.strategies.append(new_strategy)

        # Исправленный вызов: передаем словарь с buy_levels
        self.db_handler.update_strategy(strategy_id, {'buy_levels': new_strategy.buy_levels})

    def run(self, interval=60):
        """Основной цикл управления стратегиями."""
        last_sync = time.time()  # Добавляем время последней синхронизации
        try:
            while self.running:  # Используем флаг running вместо бесконечного цикла
                start_time = time.time()

                # Синхронизация с БД каждые 5 минут
                if time.time() - last_sync > 300:  # 300 секунд = 5 минут
                    with self.lock:
                        self._sync_with_db()
                    last_sync = time.time()

                # Основная обработка стратегий
                with self.lock:
                    current_strategies = self.strategies.copy()

                for strategy in current_strategies:
                    try:
                        strategy.check_conditions()
                        if strategy.id:
                            self.db_handler.update_strategy(strategy.id, {'buy_levels': strategy.buy_levels})
                    except Exception as e:
                        print(f"Ошибка в стратегии {strategy.symbol}: {e}")

                # Пауза между итерациями
                processing_time = time.time() - start_time
                sleep_time = max(interval - processing_time, 0)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nОстановка стратегий...")
            self._shutdown()
        except Exception as e:
            print(f"Критическая ошибка: {e}")
            self._shutdown()

    def _sync_with_db(self):
        """Синхронизирует список стратегий с базой данных"""
        db_strategies = self.db_handler.load_active_strategies()
        db_ids = {s['id'] for s in db_strategies}

        # Удаляем стратегии, отсутствующие в БД
        for s in self.strategies.copy():
            if s.id not in db_ids:
                print(f"Удаление неактивной стратегии {s.id}")
                self.strategies.remove(s)

        # Добавляем новые стратегии из БД
        existing_ids = {s.id for s in self.strategies}
        for data in db_strategies:
            if data['id'] not in existing_ids:
                print(f"Добавление новой стратегии {data['id']}")
                strategy = LadderStrategy.from_dict(data)
                self.strategies.append(strategy)

    def _shutdown(self):
        """Сохранение состояния при завершении работы."""
        with self.lock:
            for strategy in self.strategies:
                if strategy.id:
                    self.db_handler.update_strategy(strategy.id, {'buy_levels': strategy.buy_levels})
            self.db_handler.close()
        self.running = False