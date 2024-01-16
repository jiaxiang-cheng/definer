import numpy as np


def get_amount0(sqrtA, sqrtB, liquidity, decimals):
    # swap boundaries if not ordered
    (sqrtA, sqrtB) = (sqrtB, sqrtA) if sqrtA > sqrtB else (sqrtA, sqrtB)

    amount0 = (liquidity * 2 ** 96 * (sqrtB - sqrtA) / sqrtB / sqrtA) / 10 ** decimals

    return amount0


def get_amount1(sqrtA, sqrtB, liquidity, decimals):
    # swap boundaries if not ordered
    (sqrtA, sqrtB) = (sqrtB, sqrtA) if sqrtA > sqrtB else (sqrtA, sqrtB)

    amount1 = liquidity * (sqrtB - sqrtA) / 2 ** 96 / 10 ** decimals

    return amount1


def get_amounts(asqrt, asqrtA, asqrtB, liquidity, decimal0, decimal1):
    sqrt = (np.sqrt(asqrt * 10 ** (decimal1 - decimal0))) * (2 ** 96)
    sqrtA = np.sqrt(asqrtA * 10 ** (decimal1 - decimal0)) * (2 ** 96)
    sqrtB = np.sqrt(asqrtB * 10 ** (decimal1 - decimal0)) * (2 ** 96)

    # swap boundaries if not ordered
    (sqrtA, sqrtB) = (sqrtB, sqrtA) if sqrtA > sqrtB else (sqrtA, sqrtB)

    if sqrt <= sqrtA:

        amount0 = get_amount0(sqrtA, sqrtB, liquidity, decimal0)

        return amount0, 0

    elif sqrtB > sqrt > sqrtA:

        amount0 = get_amount0(sqrt, sqrtB, liquidity, decimal0)
        amount1 = get_amount1(sqrtA, sqrt, liquidity, decimal1)

        return amount0, amount1

    else:

        amount1 = get_amount1(sqrtA, sqrtB, liquidity, decimal1)

        return 0, amount1


def get_liquidity0(sqrtA, sqrtB, amount0, decimals):
    # swap boundaries if not ordered
    (sqrtA, sqrtB) = (sqrtB, sqrtA) if sqrtA > sqrtB else (sqrtA, sqrtB)

    liquidity = amount0 / ((2 ** 96 * (sqrtB - sqrtA) / sqrtB / sqrtA) / 10 ** decimals)

    return liquidity


def get_liquidity1(sqrtA, sqrtB, amount1, decimals):
    # swap boundaries if not ordered
    (sqrtA, sqrtB) = (sqrtB, sqrtA) if sqrtA > sqrtB else (sqrtA, sqrtB)

    liquidity = amount1 / ((sqrtB - sqrtA) / 2 ** 96 / 10 ** decimals)

    return liquidity


def get_liquidity(asqrt, asqrtA, asqrtB, amount0, amount1, decimal0, decimal1):
    sqrt = (np.sqrt(asqrt * 10 ** (decimal1 - decimal0))) * (2 ** 96)  # initial price
    sqrtA = np.sqrt(asqrtA * 10 ** (decimal1 - decimal0)) * (2 ** 96)  # lower bound of price range
    sqrtB = np.sqrt(asqrtB * 10 ** (decimal1 - decimal0)) * (2 ** 96)  # upper bound of price range

    # swap boundaries if not ordered
    (sqrtA, sqrtB) = (sqrtB, sqrtA) if sqrtA > sqrtB else (sqrtA, sqrtB)

    if sqrt <= sqrtA:  # lower than lower bound

        liquidity0 = get_liquidity0(sqrtA, sqrtB, amount0, decimal0)

        return liquidity0

    elif sqrtB > sqrt > sqrtA:  # within the price range

        liquidity0 = get_liquidity0(sqrt, sqrtB, amount0, decimal0)
        liquidity1 = get_liquidity1(sqrtA, sqrt, amount1, decimal1)
        liquidity = liquidity0 if liquidity0 < liquidity1 else liquidity1

        return liquidity

    else:  # higher than upper bound

        liquidity1 = get_liquidity1(sqrtA, sqrtB, amount1, decimal1)

        return liquidity1


def get_initial_wealth(base, dpd, decimal, SMIN, SMAX, target):

    if base == 0:  # Token 0 USDC as the base token

        sqrt0 = np.sqrt(dpd['close'].iloc[-1] * 10 ** decimal)
        dpd['price0'] = dpd['close']

    else:

        sqrt0 = np.sqrt(1 / dpd['close'].iloc[-1] * 10 ** decimal)
        dpd['price0'] = 1 / dpd['close']

    if SMIN < sqrt0 < SMAX:

        deltaL = target / ((sqrt0 - SMIN) + (((1 / sqrt0) - (1 / SMAX)) * (dpd['price0'].iloc[-1] * 10 ** decimal)))
        amount1 = deltaL * (sqrt0 - SMIN)
        amount0 = deltaL * ((1 / sqrt0) - (1 / SMAX)) * 10 ** decimal

    elif sqrt0 < SMIN:

        deltaL = target / (((1 / SMIN) - (1 / SMAX)) * (dpd['price0'].iloc[-1]))
        amount1 = 0
        amount0 = deltaL * ((1 / SMIN) - (1 / SMAX))

    else:

        deltaL = target / (SMAX - SMIN)
        amount1 = deltaL * (SMAX - SMIN)
        amount0 = 0

    return amount0, amount1, deltaL


def get_fee(dpd, base, mini, maxi, myliquidity, decimal0, decimal1, decimal):
    dpd[['ActiveLiq', 'amount0', 'amount1', 'amount0unb', 'amount1unb']] = 0

    if base == 0:

        for i, row in dpd.iterrows():

            if dpd['high'].iloc[i] > mini and dpd['low'].iloc[i] < maxi:
                dpd.iloc[i, dpd.columns.get_loc('ActiveLiq')] = (
                        (min(maxi, dpd['high'].iloc[i]) - max(dpd['low'].iloc[i], mini)) /
                        (dpd['high'].iloc[i] - dpd['low'].iloc[i]) * 100)
            else:
                dpd.iloc[i, dpd.columns.get_loc('ActiveLiq')] = 0

            amounts = get_amounts(dpd['price0'].iloc[i],
                                  mini, maxi, myliquidity, decimal0, decimal1)
            dpd.iloc[i, dpd.columns.get_loc('amount0')] = amounts[1]
            dpd.iloc[i, dpd.columns.get_loc('amount1')] = amounts[0]

            amountsunb = get_amounts((dpd['price0'].iloc[i]),
                                     1.0001 ** (-887220), 1.0001 ** 887220, 1, decimal0, decimal1)
            dpd.iloc[i, dpd.columns.get_loc('amount0unb')] = amountsunb[1]
            dpd.iloc[i, dpd.columns.get_loc('amount1unb')] = amountsunb[0]

    else:

        for i, row in dpd.iterrows():

            if (1 / dpd['low'].iloc[i]) > mini and (1 / dpd['high'].iloc[i]) < maxi:
                dpd.iloc[i, dpd.columns.get_loc('ActiveLiq')] = (min(maxi, 1 / dpd['low'].iloc[i]) - max(
                    1 / dpd['high'].iloc[i], mini)) / ((1 / dpd['low'].iloc[i]) - (1 / dpd['high'].iloc[i])) * 100
            else:
                dpd.iloc[i, dpd.columns.get_loc('ActiveLiq')] = 0

            amounts = get_amounts((dpd['price0'].iloc[i] * 10 ** decimal),
                                  mini, maxi, myliquidity, decimal0, decimal1)
            dpd.iloc[i, dpd.columns.get_loc('amount0')] = amounts[0]
            dpd.iloc[i, dpd.columns.get_loc('amount1')] = amounts[1]

            amountsunb = get_amounts((dpd['price0'].iloc[i]),
                                     1.0001 ** (-887220), 1.0001 ** 887220, 1, decimal0, decimal1)
            dpd.iloc[i, dpd.columns.get_loc('amount0unb')] = amountsunb[0]
            dpd.iloc[i, dpd.columns.get_loc('amount1unb')] = amountsunb[1]

    dpd['myfee0'] = dpd['fee0token'] * myliquidity * dpd['ActiveLiq'] / 100
    dpd['myfee1'] = dpd['fee1token'] * myliquidity * dpd['ActiveLiq'] / 100

    if base == 0:

        dpd['feeV'] = dpd['myfee0'] + dpd['myfee1'] * dpd['close']
        dpd['amountV'] = dpd['amount0'] + dpd['amount1'] * dpd['close']
        dpd['amountunb'] = dpd['amount0unb'] + dpd['amount1unb'] * dpd['close']
        dpd['fgV'] = dpd['fee0token'] + dpd['fee1token'] * dpd['close']

        dpd['feeusd'] = dpd['feeV'] * (
                dpd['pool.totalValueLockedUSD'].iloc[0] /
                (dpd['pool.totalValueLockedToken1'].iloc[0] * dpd['close'].iloc[0] + (
                dpd['pool.totalValueLockedToken0'].iloc[0]))
        )

    else:

        dpd['feeV'] = dpd['myfee0'] / dpd['close'] + dpd['myfee1']
        dpd['amountV'] = dpd['amount0'] / dpd['close'] + dpd['amount1']
        dpd['feeVbase0'] = dpd['myfee0'] + dpd['myfee1'] * dpd['close']
        dpd['amountunb'] = dpd['amount0unb'] / dpd['close'] + dpd['amount1unb']
        dpd['fgV'] = dpd['fee0token'] / dpd['close'] + dpd['fee1token']

        dpd['feeusd'] = dpd['feeV'] * (
                dpd['pool.totalValueLockedUSD'].iloc[0] /
                (dpd['pool.totalValueLockedToken1'].iloc[0] + (
                            dpd['pool.totalValueLockedToken0'].iloc[0] / dpd['close'].iloc[0]))
        )

    return dpd
