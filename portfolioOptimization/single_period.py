# Copyright 2021 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import product
import json
import random

import numpy as np
import pandas as pd
from pandas_datareader.data import DataReader
from dimod import Integer, Binary
from dimod import quicksum
from dimod import ConstrainedQuadraticModel, DiscreteQuadraticModel
from dwave.system import LeapHybridDQMSampler, LeapHybridCQMSampler


class SinglePeriod:
    """Define and solve a  single-period portfolio optimization problem.
    """
    def __init__(self, stocks=('AAPL', 'MSFT', 'AAL', 'WMT'), budget=1000,
                 bin_size=None, gamma=None, file_path='data/basic_data.csv',
                 dates=None, model_type='CQM', alpha=0.005, baseline=None,
                 sampler_args=None, t_cost=0.01, verbose=True,
                 label='PortOpt', init_holdings=None, window_size=1):
        """Class constructor.
        Args:
            stocks (list of str): List of stocks.
            budget (int): Portfolio budget.
            bin_size (int): Maximum number of intervals for each stock.
            gamma (float or int or list or tuple): Budget constraint penalty coefficient(s).
                If gamma is a tuple/list and model is DQM, grid search will be done;
                otherwise, no grid search.
            file_path (str): Full path of CSV file containing stock data.
            dates (list of str): Pair of strings for start date and end date.
            model_type (str): CQM or DQM.
            alpha (float or int or list or tuple): Risk aversion coefficient.
                If alpha is a tuple/list and model is DQM, grid search will be done;
                otherwise, no grid search.
            baseline (str): Stock baseline for rebalancing model.
            sampler_args (dict): Sampler arguments.
            t_cost (float): transaction cost; percentage of transaction dollar value.
            verbose (bool): Flag to enable additional output.
        """
        self.stocks = list(stocks)
        self.budget = budget
        self.init_budget = budget
        self.gamma_list = []
        self.file_path = file_path
        self.dates = dates
        self.model_type = model_type
        self.alpha_list = []
        self.baseline = [baseline]
        self.verbose = verbose
        self.t_cost = t_cost
        self.window_size = window_size
        if init_holdings:
            self.init_holdings = init_holdings
        else:
            self.init_holdings = {s:0 for s in self.stocks}
        self.label = label
        print(f'init_holdings: {init_holdings}')

        print(f'self.dates: {self.dates}')

        if isinstance(alpha, (list, tuple)):
            self.alpha = alpha[0]
            self.alpha_list = list(alpha)
        elif isinstance (alpha, (int, float)):
            self.alpha = alpha
        else:
            raise TypeError("Use integer or float for alpha, or a list or tuple of either type.")

        if gamma:
            if isinstance(gamma, (list, tuple)):
                self.gamma = gamma[-1]
                self.gamma_list = list(gamma)
            elif isinstance(gamma, (int, float)):
                self.gamma = gamma
            else:
                raise TypeError("Use integer or float for gamma, or a list or tuple of either type.")
        else:
            self.gamma = 100

        if bin_size:
            self.bin_size = bin_size
        else:
            self.bin_size = 10

        self.model = {'CQM': None, 'DQM': None}

        self.sample_set = {}
        if sampler_args:
            self.sampler_args = json.loads(sampler_args)
        else:
            self.sampler_args = {}

        self.sampler = {'CQM': LeapHybridCQMSampler(**self.sampler_args),
                        'DQM': LeapHybridDQMSampler(**self.sampler_args)}

        self.solution = {}

        self.precision = 2

    def load_data(self, file_path='', dates=None, df=None, num=0):
        """Load the relevant stock data from file, dataframe, or Yahoo!.
        Args:
            file_path (string): Full path of csv file containing stock price data
                for the single period problem.
            dates (list): [Start_Date, End_Date] to query data from Yahoo!.
            df (dataframe): Table of stock prices.
            num (int): Number of stocks to be randomnly generated.
        """
        if df is not None:
            print("\nLoading data from DataFrame...")
            self.df = df
            self.stocks = df.columns.tolist()
        elif file_path is not None:
            print("\nLoading data from provided CSV file...")
            self.file_path = file_path
            print(f'file: {self.file_path}')
            self.df = pd.read_csv(self.file_path, index_col=0)
            #self.df = self.df.set_index('Date')
            for column in self.df.columns:
                if column != 'Date':
                    self.df[column] = self.df[column].astype(np.float32)
                else:
                    self.df[column] = pd.to_datetime(self.df[column])
            print(f'types: {self.df.dtypes}')
            self.df_all = self.df.copy()
            #self.df_all.index = pd.to_datetime(self.df_all.index)

            self.rolling_avg = self.df_all.rolling(window=1).mean()
            self.rolling_avg.reset_index(inplace=True)
            print(f'df_all: {self.df_all.head()}')
            # Read in baseline data; resample to monthly

            if self.baseline[0] is not None:
                print(f'inside baseline: {self.baseline is None}')
                print(f'none type: {type(self.baseline)}')
                print(f'baseline: {self.baseline}')
                index_df = DataReader(self.baseline, 'yahoo',
                                  self.dates[0], self.dates[1])
                index_df = index_df.resample('BM').last()


                self.df_baseline = pd.DataFrame(index=index_df.index,
                                            columns=self.baseline)

                for i in self.baseline:
                    self.df_baseline[i] = index_df[[('Adj Close',  i)]]
                print(f"Baseline: {self.df_baseline.head()}")

        elif dates or self.dates:
            if dates:
                self.dates = dates
            print(self.dates)

            print(f"\nLoading live data from the web from Yahoo! finance",
                  f"from {self.dates[0]} to {self.dates[1]}...")

            # Generating random list of stocks
            # if num > 0:
            #     if (self.dates[0] < '2010-01-01'):
            #         raise Exception(f"Start date must be >= '2010-01-01' "
            #                         f"when using option 'num'.")
            #     symbols_df = pd.read_csv('data/stocks_symbols.csv')
            #     self.stocks = random.sample(list(symbols_df.loc[:,'Symbol']), num)

            # Read in daily data; resample to monthly
            panel_data = DataReader(self.stocks, 'yahoo',
                                    self.dates[0], self.dates[1])
            panel_data = panel_data.resample('BM').last()
            self.df_all = pd.DataFrame(index=panel_data.index,
                                       columns=self.stocks)

            for i in self.stocks:
                self.df_all[i] = panel_data[[('Adj Close',  i)]]
            self.rolling_avg = self.df_all.rolling(window=self.window_size).mean()
            self.rolling_avg.reset_index(inplace=True)
            #
            # Read in baseline data; resample to monthly
            if self.baseline[0] is not None:
                index_df = DataReader(self.baseline, 'yahoo',
                                      self.dates[0], self.dates[1])
                index_df = index_df.resample('BM').last()
                self.df_baseline = pd.DataFrame(index=index_df.index,
                                                columns=self.baseline)

                for i in self.baseline:
                    self.df_baseline[i] = index_df[[('Adj Close',  i)]]

            self.df = self.df_all
        else:
            print('No data loaded')
            exit()

        print(f'budget: {self.budget}')
        print(f'self.df.iloc[-1]: {self.df.iloc[-1]}')
        try:
            self.max_num_shares = (self.budget/self.df.iloc[-1]).astype(int)
        except Exception as e:
            print('Exception')
            print(f'{self.budget/self.df.iloc[-1]}')
            print(f'self.df.iloc[-1]: {self.df.iloc[-1].values.tolist()}')
            exit()
        if self.verbose:
            print("\nMax shares we can afford with a budget of", self.budget)
            print(self.max_num_shares.to_string())

        self.shares_intervals = {}
        for stock in self.stocks:
            if self.max_num_shares[stock]+1 <= self.bin_size:
                self.shares_intervals[stock] = list(range(self.max_num_shares[stock] + 1))
            else:
                span = (self.max_num_shares[stock]+1) / self.bin_size
                self.shares_intervals[stock] = [int(i*span)
                                                for i in range(self.bin_size)]

        print(f'shares_intervals: {self.shares_intervals}')

        self.price = self.df.iloc[-1]
        self.monthly_returns = self.df[list(self.stocks)].pct_change().iloc[1:]
        self.avg_monthly_returns = self.monthly_returns.mean(axis=0)
        self.covariance_matrix = self.monthly_returns.cov()

    def build_cqm(self, max_risk=None, min_return=None, idx=0):
        """Build and store a CQM.
        This method allows the user a choice of 3 problem formulations:
            1) max return - alpha*risk (default formulation)
            2) max return s.t. risk <= max_risk
            3) min risk s.t. return >= min_return
        Args:
            max_risk (int): Maximum risk for the risk bounding formulation.
            min_return (int): Minimum return for the return bounding formulation.
            init_holdings (float): Initial holdings, or initial portfolio state.
        """
        # Instantiating the CQM object
        cqm = ConstrainedQuadraticModel()

        # Defining and adding variables to the CQM model
        x = {s: Integer("%s" %s, lower_bound=0,
                        upper_bound=self.max_num_shares[s]) for s in self.stocks}

        print('covariance matrix')
        print(self.covariance_matrix)
        # Defining risk expression
        risk = 0
        for s1, s2 in product(self.stocks, self.stocks):
            coeff = (self.covariance_matrix[s1][s2] * self.price[s1] * self.price[s2])
            risk = risk + coeff*x[s1]*x[s2]

        print(f'min risk: {max_risk}')
        # Defining the returns expression
        print(f'price: {self.price}')
        print(f'avg_monthly_returns: {self.avg_monthly_returns}')
        returns = 0
        for s in self.stocks:
#            returns = returns + self.price[s] * self.avg_monthly_returns[s] * x[s]
            returns = returns + self.price[s] * self.rolling_avg.loc[idx, s] * x[s]

        # Adding budget and related constraints
        # if not init_holdings:
        #     init_holdings = self.init_holdings
        # else:
        #     self.init_holdings = init_holdings

        if not self.t_cost:
            print('not t_cost')
            cqm.add_constraint(quicksum([x[s]*self.price[s] for s in self.stocks])
                               <= self.budget, label='upper_budget')
            cqm.add_constraint(quicksum([x[s]*self.price[s] for s in self.stocks])
                               >= 0.997*self.budget, label='lower_budget')
        else:
            print('yes t_cost')
            print(f't_cost: {self.t_cost}')
            # Modeling transaction cost
            x0 = self.init_holdings
            print(f'initial_holdings: {x0}')

            y = {s: Binary("Y[%s]" %s) for s in self.stocks}

            lhs = 0
            for s in self.stocks:
                lhs = lhs + 2*self.t_cost*self.price[s]*x[s]*y[s] \
                      + self.price[s]*(1 - self.t_cost)*x[s] \
                      - 2*self.t_cost*self.price[s]*x0[s]*y[s] \
                      - self.price[s]*(1 - self.t_cost)*x0[s]

            print(f'lhs: {lhs}')

            cqm.add_constraint( lhs <= self.budget, label='upper_budget')
            cqm.add_constraint( lhs >= self.budget - 0.003*self.init_budget,
                                label='lower_budget')

            # indicator constraints
            for s in self.stocks:
                print(f'stock {s}')
                cqm.add_constraint(x[s] - x0[s]*y[s] >= 0,
                                   label=f'indicator_constraint_gte_{s}')
                print(f'indicator_constraint_gte: {x[s] - x0[s]*y[s]} >= 0')
                cqm.add_constraint(x[s] - x[s]*y[s] <= x0[s],
                                   label=f'indicator_constraint_lte_{s}')
                print(f'indicator_constraint_lte: {x[s] - x[s]*y[s]} < {x0[s]}')

        if max_risk:
            # Adding maximum risk constraint
            cqm.add_constraint(risk <= max_risk, label='max_risk')

            # Objective: maximize return
            cqm.set_objective(-1*returns)
        elif min_return:
            # Adding minimum returns constraint
            cqm.add_constraint(returns >= min_return, label='min_return')

            # Objective: minimize risk
            cqm.set_objective(risk)
        else:
            # Objective: minimize mean-variance expression
            cqm.set_objective(self.alpha*risk - returns)

        cqm.substitute_self_loops()

        self.model['CQM'] = cqm

    def solve_cqm(self, max_risk=None, min_return=None,  idx=0):
        """Solve CQM.
        This method allows the user to solve one of 3 cqm problem formulations:
            1) max return - alpha*risk (default formulation)
            2) max return s.t. risk <= max_risk
            3) min risk s.t. return >= min_return
        Args:
            max_risk (int): Maximum risk for the risk bounding formulation.
            min_return (int): Minimum return for the return bounding formulation.
            init_holdings (float): Initial holdings, or initial portfolio state.
        Returns:
            solution (dict): This is a dictionary that saves solutions in desired format
                e.g., solution = {'stocks': {'IBM': 3, 'WMT': 12}, 'risk': 10, 'return': 20}
        """
        self.build_cqm(max_risk, min_return, idx=idx)

        self.sample_set['CQM'] = self.sampler['CQM'].sample_cqm(self.model['CQM'], label=self.label)
        n_samples = len(self.sample_set['CQM'].record)
        print(f'n_samples: {n_samples}')

        print(f'type sample_set: {type(self.sample_set["CQM"])}')
        print('sample_set')
        print(self.sample_set['CQM'])
        f = open("sample_set.txt","w")
        f.write(str(self.sample_set))
        f.close()


        feasible_samples = self.sample_set['CQM'].filter(lambda d: d.is_feasible)
        print(f'type feasible_samples: {type(feasible_samples)}')
        print(f'feasible_samples: {feasible_samples}')
        f = open("feasible_samples.txt","w")
        f.write( str(feasible_samples) )
        f.close()


        if not feasible_samples:
            raise Exception("No feasible solution could be found for this problem instance.")
        else:
            best_feasible = feasible_samples.first

            solution = {}

            solution['stocks'] = {k:int(best_feasible.sample[k]) for k in self.stocks}
            print(f'solution_stocks: {solution["stocks"]}')

            solution['return'], solution['risk'] = self.compute_risk_and_returns(solution['stocks'])

            print(f'solution_return: {solution["return"]}')

            print(f'solution_risk: {solution["risk"]}')

            spending = sum([self.price[s]*max(0, solution['stocks'][s] - self.init_holdings[s]) for s in self.stocks])
            print(f'spending: {spending}')
            sales = sum([self.price[s]*max(0, self.init_holdings[s] - solution['stocks'][s]) for s in self.stocks])
            print(f'sales: {sales}')

            transaction = self.t_cost*(spending + sales)
            print(f'transaction: {transaction}')

            if self.verbose:
                print(f'Number of feasible solutions: {len(feasible_samples)} out of {n_samples} sampled.')
                print(f'\nBest energy: {self.sample_set["CQM"].first.energy: .2f}')
                print(f'Best energy (feasible): {best_feasible.energy: .2f}')

            print(f'\nBest feasible solution:')
            print("\n".join("{}\t{:>3}".format(k, v) for k, v in solution['stocks'].items()))

            print(f"\nEstimated Returns: {solution['return']}")

            print(f"Sales Revenue: {sales:.2f}")

            print(f"Purchase Cost: {spending:.2f}")

            print(f"Transaction Cost: {transaction:.2f}")

            print(f"Variance: {solution['risk']}\n")

            return solution

    def build_dqm(self, alpha=None, gamma=None):
        """Build DQM.
        Args:
            alpha (float): Risk aversion coefficient.
            gamma (int): Penalty coefficient for budgeting constraint.
        """
        if gamma:
            self.gamma = gamma

        if alpha:
            self.alpha = alpha

        # Defining DQM
        dqm = DiscreteQuadraticModel()

        # Build the DQM starting by adding variables
        for s in self.stocks:
            dqm.add_variable(len(self.shares_intervals[s]), label=s)

        # Objective 1: minimize variance
        for s1, s2 in product(self.stocks, self.stocks):
            coeff = (self.covariance_matrix[s1][s2]
                     * self.price[s1] * self.price[s2])
            if s1 == s2:
                for k in range(dqm.num_cases(s1)):
                    num_s1 = self.shares_intervals[s1][k]
                    dqm.set_linear_case(
                        s1, k,
                        dqm.get_linear_case(s1,k)
                        + self.alpha*coeff*num_s1*num_s1)
            else:
                for k in range(dqm.num_cases(s1)):
                    for m in range(dqm.num_cases(s2)):
                        num_s1 = self.shares_intervals[s1][k]
                        num_s2 = self.shares_intervals[s2][m]

                        dqm.set_quadratic_case(
                            s1, k, s2, m,
                            dqm.get_quadratic_case(s1,k,s2,m)
                            + coeff*self.alpha*num_s1*num_s2)

                        # Objective 2: maximize return
        for s in self.stocks:
            for j in range(dqm.num_cases(s)):
                dqm.set_linear_case(
                    s, j, dqm.get_linear_case(s,j)
                          - self.shares_intervals[s][j]*self.price[s]
                          * self.avg_monthly_returns[s])

        # Scaling factor to guarantee that all coefficients are integral
        # needed in order to use add_linear_inequality_constraint method
        factor = 10**self.precision

        min_budget = round(factor*0.997*self.budget)
        budget = int(self.budget)

        terms = [(s, j, int(self.shares_intervals[s][j]
                            *factor*self.price[s]))
                 for s in self.stocks
                 for j in range(dqm.num_cases(s))]

        dqm.add_linear_inequality_constraint(terms,
                                             constant=0,
                                             lb=min_budget,
                                             ub=factor*budget,
                                             lagrange_multiplier=self.gamma,
                                             label="budget")

        self.model['DQM'] = dqm

    def solve_dqm(self):
        """Solve DQM.
        Returns:
            solution (dict): This is a dictionary that saves solutions in desired format
                e.g., solution = {'stocks': {'IBM': 3, 'WMT': 12}, 'risk': 10, 'return': 20}
        """
        if not self.model['DQM']:
            self.build_dqm()

        self.sample_set['DQM'] = self.sampler['DQM'].sample_dqm(self.model['DQM'],
                                                                label="Example - Portfolio Optimization")

        solution = {}

        sample = self.sample_set['DQM'].first.sample
        solution['stocks'] = {s:self.shares_intervals[s][sample[s]] for s in self.stocks}

        solution['return'], solution['risk'] = self.compute_risk_and_returns(solution['stocks'])

        spending = sum([self.price[s]*solution['stocks'][s] for s in self.stocks])

        print(f'\nDQM -- solution for alpha == {self.alpha} and gamma == {self.gamma}:')
        print(f"\nShares to buy:")

        print("\n".join("{}\t{:>3}".format(k, v) for k, v in solution['stocks'].items()))

        print(f"\nEstimated returns: {solution['return']}")

        print(f"Purchase Cost: {spending:.2f}")

        print(f"Variance: {solution['risk']}\n")


        return solution

    def dqm_grid_search(self):
        """Execute parameter (alpha, gamma) grid search for DQM.
        """
        alpha = self.alpha_list
        gamma = self.gamma_list

        data_matrix = np.zeros((len(alpha), len(gamma)))

        if self.verbose:
            print("\nGrid search results:")

        for i in range(len(alpha)):
            for j in range(len(gamma)):

                alpha_i = alpha[i]
                gamma_j = gamma[j]

                self.build_dqm(alpha_i, gamma_j)

                # Solve the problem using the DQM solver
                solution = self.solve_dqm()

                data_matrix[i,j] = solution['return'] / np.sqrt(solution['risk'])

        n_opt = np.argmax(data_matrix)

        self.alpha = alpha[n_opt//len(gamma)]
        self.gamma = gamma[n_opt - (n_opt//len(gamma)) * len(gamma)]

        print(f"DQM Grid Search Completed: alpha={self.alpha}, gamma={self.gamma}.-")

    def compute_risk_and_returns(self, solution):
        """Compute the risk and return values of solution.
        """
        variance = 0.0
        print(f'covariance_matrix: {self.covariance_matrix}')
        for s1, s2 in product(solution, solution):
            print(s1,s2)
            variance += (solution[s1] * self.price[s1]
                         * solution[s2] * self.price[s2]
                         * self.covariance_matrix[s1][s2])

        est_return = 0
        for stock in solution:
            est_return += solution[stock]*self.price[stock]*self.avg_monthly_returns[stock]

        return round(est_return, 2), round(variance, 2)

    def run(self, min_return=0, max_risk=0, num=0):
        """Execute sequence of load_data --> build_model --> solve.
        Args:
            max_risk (int): Maximum risk for the risk bounding formulation.
            min_return (int): Minimum return for the return bounding formulation.
            num (int): Number of stocks to be randomnly generated.
            init_holdings (float): Initial holdings, or initial portfolio state.
        """
        self.load_data(num=num)
        if self.model_type=='CQM':
            print(f"\nCQM run...")
            self.solution['CQM'] = self.solve_cqm(min_return=min_return,
                                                  max_risk=max_risk,
                                                  init_holdings=self.init_holdings)
        else:
            print(f"\nDQM run...")
            if len(self.alpha_list) > 1 or len(self.gamma_list) > 1:
                print("\nStarting DQM Grid Search...")
                self.dqm_grid_search()

            self.build_dqm()
            self.solution['DQM'] = self.solve_dqm()