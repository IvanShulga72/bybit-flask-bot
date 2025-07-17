# extensions.py
from flask import Flask
from pybit.unified_trading import HTTP
from keys import api, secret

app = Flask(__name__)
db_handler = None  # Будет инициализировано позже
bybit_session = HTTP(testnet=False, api_key=api, api_secret=secret)