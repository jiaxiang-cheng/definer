import requests


class Derivative:
    def __init__(self, name, payoff_fun, bid, ask, info):
        self.name = name
        self.payoff_fun = payoff_fun
        self.deribit_info = info
        self.bid = bid
        self.ask = ask


def retrieve_deribit_options(currency, expired=False):
    assert currency in ["ETH", "SOL", "BTC"]
    expired = "true" if expired else "false"
    instruments = requests.get(
        "https://deribit.com/api/v2/public/get_book_summary_by_currency?currency={}&kind=option".format(currency)
    ).json()["result"]
    return instruments