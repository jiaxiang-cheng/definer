import math
import numpy as np
from scipy.stats import norm
import requests


class Derivative:
    def __init__(self, name, payoff_fun, bid, ask, info):
        self.name = name
        self.payoff_fun = payoff_fun
        self.deribit_info = info
        self.bid = bid
        self.ask = ask


def retrieve_deribit_options(currency, expired=False):
    assert currency in ["ETH", "BTC", "USDC", "USDT"]
    # expired = "true" if expired else "false"

    # https://docs.deribit.com/#public-get_book_summary_by_currency
    instruments = requests.get(
        "https://deribit.com/api/v2/public/get_book_summary_by_currency?currency={}&kind=option".format(currency)
    ).json()["result"]

    return instruments


def black_scholes(df_options, initial_price, r, sigma):
    df_options['pricing'] = 0
    df_options['d1'] = (np.log(initial_price / df_options.strike_price.astype('float')) + (r + 0.5 * sigma ** 2) *
                        df_options['T']) / (sigma * np.sqrt(df_options['T']))
    df_options['d2'] = (np.log(initial_price / df_options.strike_price.astype('float')) + (r - 0.5 * sigma ** 2) *
                        df_options['T']) / (sigma * np.sqrt(df_options['T']))
    df_options.loc[df_options['option_type'] == 'C', 'pricing'] = (
            (initial_price * norm.cdf(df_options['d1']) -
             df_options.strike_price.astype('float') * np.exp(-r * df_options['T']) * norm.cdf(df_options['d2']))
            / initial_price)
    df_options.loc[df_options['option_type'] == 'P', 'pricing'] = (
            (df_options.strike_price.astype('float') * np.exp(-r * df_options['T']) * norm.cdf(-df_options['d2']) -
             initial_price * norm.cdf(-df_options['d1']))
            / initial_price)
    return df_options
