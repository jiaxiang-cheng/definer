# DeFiNER
**Decentralized Finance Navigates Every Route**    
A Solution Framework for Modeling and Hedging Impermanent Loss and
Dynamic Liquidity Provision Using Deep Reinforcement Learning
in Uniswap V3 with Concentrated Liquidity. *Fintech-As-A-Service: Hackathon of NUS Fintech Summit 2024*.     
Authors: 
[Jiaxiang Cheng](https://www.linkedin.com/in/jiaxiang-cheng/),
[Xuejing Lyu](https://www.linkedin.com/in/xuejing-lyu-752297178/)

-----

This project provides a solution framework for hedging impermanent loss of liquidity
providers in Uniswap V3 with concentrated liquidity provision, which is realized
with a delta-gamma hedging strategy. The bact-testing showcase is presented below:

![showcase](img/viz_back-testing.png)

[//]: # (![showcase]&#40;img/back-testing_delta-gamma.png&#41;)

Based on the back-testing results, 
of which the process is documented in `3.3 Back-testing Delta-Gamma Hedge.ipynb` in detail,
the proposed hedging strategy is potential for
fully hedging the impermanent loss raised from the liquidity position in Uniswap V3.


[//]: # (```)

[//]: # (pip install "fastapi[all]")

[//]: # (```)

[//]: # (```)

[//]: # (uvicorn main:app --reload)

[//]: # (```)
[//]: # (http://127.0.0.1:8000    )
[//]: # (http://127.0.0.1:8000/docs)