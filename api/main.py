from fastapi import FastAPI

import numpy as np
from datetime import datetime
from definer.data import graph

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/latest-price")
async def get_latest_price():
    """
    This API retrieve latest price from The Graph with poolHourDatas
    """
    address = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
    after = '2022-01-01'
    dpd = graph(address, int(datetime.timestamp(datetime.strptime(after, "%Y-%m-%d"))))

    return {
        "latest price": dpd['close'].iloc[0],
        "starting time": datetime.utcfromtimestamp(dpd.periodStartUnix.min()).strftime('%Y-%m-%d %H:%M:%S'),
        "ending time": datetime.utcfromtimestamp(dpd.periodStartUnix.max()).strftime('%Y-%m-%d %H:%M:%S'),
    }


@app.get("/impermanent-loss")
async def impermanent_loss(
        lower_price: float,
        upper_price: float,
):
    """
    This API run KM for each brand in brands and return results
    """
    address = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
    after = '2022-01-01'
    dpd = graph(address, int(datetime.timestamp(datetime.strptime(after, "%Y-%m-%d"))))

    latest_price = dpd['close'].iloc[0]
    pu = upper_price / latest_price
    pl = lower_price / latest_price

    x1 = np.linspace(0, pl, 100)
    y1 = (np.sqrt(pl) + x1 / np.sqrt(pl) - 1 - x1) / (1 + x1 - np.sqrt(pl) - x1 / np.sqrt(pu))

    x2 = np.linspace(pl, pu, 100)
    y2 = (2 * np.sqrt(x2) - 1 - x2) / (1 + x2 - np.sqrt(pl) - x2 / np.sqrt(pu))

    x3 = np.linspace(pu, 5, 100)
    y3 = (x3 / np.sqrt(pu) + np.sqrt(pu) - 1 - x3) / (1 + x3 - np.sqrt(pl) - x3 / np.sqrt(pu))

    il_list = []
    for index, (x, y) in enumerate(zip(x1, y1)):
        il_list.append({'x': x, 'IL': y})
    for index, (x, y) in enumerate(zip(x2, y2)):
        il_list.append({'x': x, 'IL': y})
    for index, (x, y) in enumerate(zip(x3, y3)):
        il_list.append({'x': x, 'IL': y})

    return {
        "data": {
            "latest price": latest_price,
        },
        "results": il_list,
    }
