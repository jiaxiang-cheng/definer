import pandas as pd
from datetime import datetime

from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport


def graph(address, after):

    sample_transport = RequestsHTTPTransport(
        url='https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3',
        retries=5
    )

    client = Client(
        transport=sample_transport
    )

    # print(datetime.utcfromtimestamp(after).strftime('%Y-%m-%d %H:%M:%S'))

    query = gql(
        '''
        query ($after: Int!){
            poolHourDatas(
                where: {pool: "''' + str(address) + '''", periodStartUnix_gt: $after},
                orderBy: periodStartUnix,
                orderDirection: desc,
                first: 1000
            ) {
                periodStartUnix
                liquidity
                high
                low
                pool {
                    totalValueLockedUSD
                    totalValueLockedToken1
                    totalValueLockedToken0
                    token0 {
                        decimals
                    }
                    token1 {
                        decimals
                    }
                }
                close
                feeGrowthGlobal0X128
                feeGrowthGlobal1X128
            }
        }
        '''
    )

    params = {
        "after": after
    }

    response = client.execute(query, variable_values=params)
    dpd = pd.json_normalize(response['poolHourDatas'])
    dpd = dpd.astype(float)
    return dpd