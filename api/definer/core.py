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


def get_initial_wealth(base, df, decimal, lower, upper, target):

    if base == 0:  # Token 0 USDC as the base token

        sqrt0 = np.sqrt(df['close'].iloc[-1] * 10 ** decimal)
        df['price0'] = df['close']

    else:

        sqrt0 = np.sqrt(1 / df['close'].iloc[-1] * 10 ** decimal)
        df['price0'] = 1 / df['close']

    if lower < sqrt0 < upper:

        deltaL = target / ((sqrt0 - lower) + (((1 / sqrt0) - (1 / upper)) * (df['price0'].iloc[-1] * 10 ** decimal)))
        amount1 = deltaL * (sqrt0 - lower)
        amount0 = deltaL * ((1 / sqrt0) - (1 / upper)) * 10 ** decimal

    elif sqrt0 < lower:

        deltaL = target / (((1 / lower) - (1 / upper)) * (df['price0'].iloc[-1]))
        amount1 = 0
        amount0 = deltaL * ((1 / lower) - (1 / upper))

    else:

        deltaL = target / (upper - lower)
        amount1 = deltaL * (upper - lower)
        amount0 = 0

    return amount0, amount1, deltaL


def get_fee(df, base, mini, maxi, liquidity, decimal0, decimal1, decimal):
    df[['ActiveLiq', 'amount0', 'amount1', 'amount0unb', 'amount1unb']] = 0

    if base == 0:

        for i, row in df.iterrows():

            if df['high'].iloc[i] > mini and df['low'].iloc[i] < maxi:
                df.iloc[i, df.columns.get_loc('ActiveLiq')] = (
                        (min(maxi, df['high'].iloc[i]) - max(df['low'].iloc[i], mini)) /
                        (df['high'].iloc[i] - df['low'].iloc[i]) * 100)
            else:
                df.iloc[i, df.columns.get_loc('ActiveLiq')] = 0

            amounts = get_amounts(df['price0'].iloc[i],
                                  mini, maxi, liquidity, decimal0, decimal1)
            df.iloc[i, df.columns.get_loc('amount0')] = amounts[1]
            df.iloc[i, df.columns.get_loc('amount1')] = amounts[0]

            amountsunb = get_amounts((df['price0'].iloc[i]),
                                     1.0001 ** (-887220), 1.0001 ** 887220, 1, decimal0, decimal1)
            df.iloc[i, df.columns.get_loc('amount0unb')] = amountsunb[1]
            df.iloc[i, df.columns.get_loc('amount1unb')] = amountsunb[0]

    else:

        for i, row in df.iterrows():

            if (1 / df['low'].iloc[i]) > mini and (1 / df['high'].iloc[i]) < maxi:
                df.iloc[i, df.columns.get_loc('ActiveLiq')] = (min(maxi, 1 / df['low'].iloc[i]) - max(
                    1 / df['high'].iloc[i], mini)) / ((1 / df['low'].iloc[i]) - (1 / df['high'].iloc[i])) * 100
            else:
                df.iloc[i, df.columns.get_loc('ActiveLiq')] = 0

            amounts = get_amounts((df['price0'].iloc[i] * 10 ** decimal),
                                  mini, maxi, liquidity, decimal0, decimal1)
            df.iloc[i, df.columns.get_loc('amount0')] = amounts[0]
            df.iloc[i, df.columns.get_loc('amount1')] = amounts[1]

            amountsunb = get_amounts((df['price0'].iloc[i]),
                                     1.0001 ** (-887220), 1.0001 ** 887220, 1, decimal0, decimal1)
            df.iloc[i, df.columns.get_loc('amount0unb')] = amountsunb[0]
            df.iloc[i, df.columns.get_loc('amount1unb')] = amountsunb[1]

    df['myfee0'] = df['fee0token'] * liquidity * df['ActiveLiq'] / 100
    df['myfee1'] = df['fee1token'] * liquidity * df['ActiveLiq'] / 100

    if base == 0:

        df['feeV'] = df['myfee0'] + df['myfee1'] * df['close']
        df['amountV'] = df['amount0'] + df['amount1'] * df['close']
        df['amountunb'] = df['amount0unb'] + df['amount1unb'] * df['close']
        df['fgV'] = df['fee0token'] + df['fee1token'] * df['close']

        df['feeusd'] = df['feeV'] * (
                df['pool.totalValueLockedUSD'].iloc[0] /
                (df['pool.totalValueLockedToken1'].iloc[0] * df['close'].iloc[0] + (
                df['pool.totalValueLockedToken0'].iloc[0]))
        )

    else:

        df['feeV'] = df['myfee0'] / df['close'] + df['myfee1']
        df['amountV'] = df['amount0'] / df['close'] + df['amount1']
        df['feeVbase0'] = df['myfee0'] + df['myfee1'] * df['close']
        df['amountunb'] = df['amount0unb'] / df['close'] + df['amount1unb']
        df['fgV'] = df['fee0token'] / df['close'] + df['fee1token']

        df['feeusd'] = df['feeV'] * (
                df['pool.totalValueLockedUSD'].iloc[0] /
                (df['pool.totalValueLockedToken1'].iloc[0] + (
                            df['pool.totalValueLockedToken0'].iloc[0] / df['close'].iloc[0]))
        )

    return df
