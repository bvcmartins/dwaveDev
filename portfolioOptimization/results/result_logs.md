### Run 1: Single-period

Single period portfolio optimization run...

Loading data from provided CSV file...

CQM run...

Best feasible solution:
AAPL	 45
MSFT	  0
AAL	  1
WMT	 10

Estimated Returns: 14.37
Sales Revenue: 0.00
Purchase Cost: 998.43
Transaction Cost: 0.00
Variance: 1373.85

qpu_access_time
00 h : 00 m : 00.015 s
charge_time
00 h : 00 m : 04.949 s
run_time
00 h : 00 m : 04.949 s

---
### Run 2: Multi-period (with rebalance)


---
### Run 3 - Single period

Single period portfolio optimization run...

Loading data from provided CSV file...

CQM run...

Best feasible solution:
XIC.TO    0
XUH.TO    0
XQQ.TO    2
VAB.TO    9
XGD.TO    6
XEI.TO   17

Estimated Returns: 11.89
Sales Revenue: 0.00
Purchase Cost: 998.29
Transaction Cost: 0.00
Variance: 1565.45

---
### Run 4: Multi-period
ETFs
---
### Run 5: failed
---
### Run 6: Test of multi-period with dataframe saving. Failed.

Loading live data from the web from Yahoo! finance from 2010-01-01 to 2010-04-01...

Date: 2010-04-30 00:00:00

Loading data from DataFrame...

Multi-Period CQM Run...

Best feasible solution:
AAPL    101
MSFT      0
AAL      39
WMT       0

Estimated Returns: 88.25
Sales Revenue: 0.00
Purchase Cost: 999.85
Transaction Cost: 0.00
Variance: 5663.13

         Date       Value AAPL MSFT AAL WMT  Variance  Returns
1  2010-04-30  999.847365  101    0  39   0   5663.13    88.25

Run completed.

---
### Run 7: complete multi-period
  - dataframe saving
  - 5 periods.
  - main run for analysis
---
### Run 8
  - multi-period
  - stocks
  - from 2010-01-01 to 2012-12-31
  - t_cost = 0.0
---
### Run 9
  - multi-period
  - stocks
  - from 2010-01-01 to 2012-12-31
  - t_cost = 0.01
---
### Run 10
  - single period
  - 2022-04-30
  - 20 stocks
  - t_cost = 0.01
---
### Run 11
  - single period
  - 20 stocks
  - min_return = 100
  - Impossible to solve
---
### Run 12
  - single period
  - 20 stocks
  - max_risk = 1000
  - Impossible to solve
---
### Run 13
  - objective: minimize variance by choosing stocks that are least correlated
  - single period
  - 2022-04-30
  - 6 stocks
  - t_cost = 0.01
---
### Run 14
  - objective: calculate 4 stocks to allow for BF checking
  - single period
  - 2022-04-30
  - 4 stocks
  - t_cost = 0.01

```
Single period portfolio optimization run...

Loading live data from the web from Yahoo! finance from 2019-01-01 to 2022-04-30...

CQM run...
min risk: 0.0
spending: 988.0440673828125
sales: 0.0

Best feasible solution:
AAPL      1
MSFT      3
AAL       0
WMT       0

Estimated Returns: 29.5
Sales Revenue: 0.00
Purchase Cost: 988.04
Transaction Cost: 9.88
Variance: 3780.49
```
---
### Run 15
  - objective: consistency check
  - same as run 14
  - single period
  - 2022-04-30
  - 4 stocks
  - t_cost = 0.01
  - conclusion
    - more verbose to check why BF is failing
    - result was consistent
---
### Run 16
  - objective: check effect of `t_cost = 0.0`
  - single period
  - 2022-04-30
  - 4 stocks
  - t_cost = 0.00
---
### Run 17
  - objective: check consistency of ExactQMSolver
  - ExactQMSolver
  - single period
  - 2022-04-30
  - 4 stocks
  - t_cost = 0.00

```
self.sample_set['CQM'] = self.sampler['Exact'].sample_cqm(self.model['CQM'], label="Portfolio Opt")

spending: 999.6610565185547
sales: 0.0

Best feasible solution:
AAPL      2
MSFT      0
AAL       4
WMT       4
```
---
### Run 18
  - objective: preparation for multi-period
  - single Period
  - 2022-05-31
  - 100 stocks
---
### Run 19
  - objective: 100 stocks multi-period
  - multi-period
  - ['2021-01-01', '2022-05-31'
  - 100 stocks
  - t_cost = 0.01
  - conclusion
    - worse than baseline (4 stocks)
---
### Run 20
  - objective: 4 stocks with recent data
  - multi-period
  - ['2021-01-01', '2022-05-31']
  - 4 stocks
  - t_cost = 0.01
  - log file was lost
  - conclusion
    - below baseline
---
### Run 21: Multi-period 4 stocks ['2018-10-01', '2020-01-01'] t_cost = 0.01
  * above baseline
---
### Run 22: Multi-period 50 random stocks ['2018-10-01', '2020-01-01'] t_cost = 0.01
  * below baseline
---
### Run 23: Multi-period 20 selected stocks ['2018-10-01', '2020-01-01'] t_cost = 0.01
  * below baseline
---
## Beginning of Developer Plan

### Run 24
  * 50 stocks - top 50 stocks in pct-change
  * ['2018-10-01', '2020-01-01']
  * global average
    - `returns = returns + self.price[s] * self.avg_monthly_returns[s] * x[s]`
  * parameters in config.yml
  * slightly above baseline
---
### Run 25
  * similar to run 24
  * ['2018-10-01', '2020-01-01']
  * 50 top stocks
  * rolling_avg
    - `returns = returns + self.price[s] * self.rolling_avg.loc[idx, s] * x[s]`
  * highly above baseline
---
### Run 26
  * 50 random stocks
  * ['2018-10-01', '2020-01-01']
  * rolling avg
  * below baseline
---
### Run 27
  * mistake
  * 50 stocks selected for ['2021-01-01', '2022-05-01'] ran instead for ['2018-10-01', '2020-01-01']
  * below baseline
---
### Run 28
  * 50 top stocks
  * ['2021-01-01', '2022-05-01']
  * rolling avg
  * checking effects of market downturn
  * below baseline in the end
---
### Run 29
  * full S&P 500 (493 stocks excluding the ones with NaNs during the period)
  * ['2018-10-01', '2020-01-01']
  * loaded from file_path
  * below baseline
---
### Run 30
  *  full S&P 500 (493 stocks excluding the ones with NaNs during the period)
  * ['2018-10-01', '2020-01-01']
  * loaded from file_path
  * Initial budget = 77,000
  * generated initial portfolio using volumes on 2018-10-31
  * result must be scaled to compare with GSPC
---
### Run 31
  * similar to run 25
  * ['2018-10-01', '2020-01-01']
  * 50 top stocks
  * budget = 100,000
  * no initial portfolio
---
### Run 32
  * 100 top stocks
  * ['2018-10-01', '2020-01-01']
  * no initial portfolio
  * budget = 1000
---

Questions

- Portfolio optimization
- got results
- need benchmarks and validation

1. What is the price model (per time)?
2. Is there a limit in queries per hour?
3. Are there tiers?
4. What is the maximum number of degrees of freedom that pegasus can handle?
5. Details about portfolio optimization:
  * why are the variances so large?
  * why cov() instead of corr() for risk calculation?
  * is there any procedure to check that the solution is a valid solution
      - ExactSolver? But it cannot optimize with constraints
6. What is the Boltzmann machine current stage of development?
7. Other than Ocean, are there low-level tools allowing for the fine tuning of lattice coupling?
8. Is there any benchmarking for Portfolio Optimization - other than the Multiverse / BBVA / Bankia work?
9. any chance I can extend the June 13th deadline? I am developing code to back the results generated.

Notes:

The notation CN refers to a Chimera graph consisting of an grid of unit cells. The D-Wave 2000Q QPU supports a C16 Chimera graph: its 2048 qubits are logically mapped into a matrix of unit cells of 8 qubits.

---

Question

Hello all,

I am studying the portfolio optimization demo posted here: https://github.com/dwave-examples/portfolio-optimization. I ran the code considering 3 portfolio scenarios: 
  1. stocks = ['AAPL', 'WMT', 'AAL', 'MSFT'] - same as demo
  2. 20 selected S&P 500 stocks including the 4 from item 1
  3. 50 random S&P 500 stocks including the 4 from item 1

Here are the common parameters for all jobs:

  * dates between 2018-10-01 and 2020-01-01 (monthly)
  * t_cost = 0.01
  * alpha = 0.005
  * budget = 1000

Results go attached. 

My question: why the 20 and 50 stocks portfolios performed so poorly when compare to 4 stocks and baseline? My understanding of the problem was that, since the 4 initial stocks were included in scenarios 2 and 3, worst case scenario the optimizer would recover the result obtained in scenario 1. Checking the Hamiltonian I didn't identify a penalty term for portfolio concentration in a few stocks - there is no reason why just a subset of 4 could be selected from 20 or 50. What am I missing here? 

Is there any parameter tuning needed to be done for these cases? 20 stocks should not in principle be such a heavy problem for Pegasus.

Thank you!
