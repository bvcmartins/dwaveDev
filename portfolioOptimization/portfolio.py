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

import click
import time
import yaml

from multi_period import MultiPeriod
from single_period import SinglePeriod

# @click.command()
# #default=['XIC.TO', 'XUH.TO', 'XQQ.TO', 'VAB.TO', 'XGD.TO', 'XEI.TO']
# @click.option('-s', '--stocks', multiple=True, type=str,
#               default=['AAPL', 'MSFT', 'AAL', 'WMT'], show_default=True,
#               help='Stock name to be included.'
#                    'When a file is provided, stock name must be included in the file ')
# @click.option('-b', '--budget', default=1000, show_default=True,
#               help='Portfolio budget')
# @click.option('-n', '--bin-size', type=int,
#               help='Maximum number of intervals for each stock. This a DQM-only option.')
# @click.option('-g', '--gamma', multiple=True, type=float,
#               help='Penalty coefficient for budget constraint. This is a DQM-only option.')
# @click.option('-a', '--alpha', multiple=True, type=float,
#               default=[0.005], show_default=True,
#               help='Risk aversion coefficient')
# @click.option('-f', '--file-path', default='data/basic_data.csv',
#               show_default=True, type=str,
#               help='Full path of csv file containing input stock data')
# @click.option('-z', '--baseline', default='^GSPC', show_default=True,
#               help='Baseline stock for comparison in multi-period run')
# @click.option('-u', '--max-risk', default=0.0,
#               help='Upper bound on risk/variance. This only works for CQM.')
# @click.option('-l', '--min-return', default=0.0,
#               help='Lower bound on the returns. This only works for CQM.')
# @click.option('-d', '--dates', nargs=2, type=str,
#               help='Start and end date to query stock data from Yahoo! Finance')
# @click.option('-m', '--model-type', default='CQM', multiple=False,
#               type=click.Choice(['CQM', 'DQM'], case_sensitive=False),
#               show_default=True, help='Model type, CQM or DQM')
# @click.option('-r', '--rebalance', is_flag=True, default=False,
#               help='Make a multi-period rebalancing portfolio optimization run; '
#                    'otherwise, make a single-period portfolio optimization run')
# @click.option('-p', '--params', default="{}",
#               help='Pass sampler arguments such as profile and solver; '
#                    'usage: -p \'{"profile": "test"}\'')
# @click.option('-v', '--verbose', is_flag=True,
#               help='Enable additional program console output')
# @click.option('-k', '--num', default=0, type=click.IntRange(0, ),
#               help='Number of stocks to be randomnly generated.'
#                    'When this option is selected, dates need to be provided.')
# @click.option('-t', '--t-cost', default=0.00, type=click.FloatRange(0, 1),
#               help='Transaction cost: percentage of transaction dollar value.')
#def main(stocks, budget, bin_size, gamma, params, file_path, max_risk, num,
#         min_return,baseline, dates, model_type, rebalance, alpha, verbose,
#         t_cost):
def main(*args, **params):

    max_risk = params['max_risk']
    min_return = params['min_return']
    model_type = params['model_type']
    gamma = params['gamma']
    bin_size = params['bin_size']
    num = params['num']
    t_cost = params['t_cost']
    stocks = params['stocks']
    budget = params['budget']
    sampler_args = params['sampler_args']
    bin_size = params['bin_size']
    dates = params['dates'],
    file_path = params['file_path']
    gamma = params['gamma'],
    model_type = params['model_type']
    alpha = params['alpha']
    verbose = params['verbose']
    baseline = params['baseline']
    label = params['label']
    init_holdings = params['init_holdings']
    window_size = params['window_size']

    if ((max_risk or min_return) and model_type != 'CQM'):
        raise Exception("The bound options require a CQM.")

#    if ((gamma or bin_size) and model_type != 'DQM'):
#        raise Exception("The option gamma or bin-size requires a DQM.")

    if (num and not dates):
        raise Exception("User must provide dates with option 'num'.")

    if (t_cost and model_type != 'CQM'):
        raise Exception("The transaction cost option requires a CQM. " \
                        "Set t_cost=0 for DQM.")

    # print(f'dates: {params["dates"]}')
    # print(f'params: {params}')

    if params['rebalance']:
        print(f"\nRebalancing portfolio optimization run...")

        my_portfolio = MultiPeriod(stocks=params["stocks"],
                                   budget=params['budget'],
                                   sampler_args=params['sampler_args'],
                                   bin_size=params['bin_size'],
                                   dates=params["dates"],
                                   file_path=params['file_path'],
                                   gamma=params['gamma'],
                                   model_type=params['model_type'],
                                   alpha=params['alpha'],
                                   verbose=params['verbose'],
                                   baseline=params['baseline'],
                                   t_cost=params['t_cost'],
                                   label=params['label'],
                                   init_holdings=params['init_holdings'],
                                   window_size=params['window_size'])
    else:
        print(f"\nSingle period portfolio optimization run...")

        my_portfolio = SinglePeriod(stocks=params['stocks'],
                                    budget=params['budget'],
                                    bin_size=params['bin_size'],
                                    gamma=params['gamma'],
                                    file_path=params['file_path'],
                                    dates=params['dates'],
                                    model_type=params['model_type'],
                                    alpha=params['alpha'],
                                    verbose=params['verbose'],
                                    sampler_args=params['sampler_args'],
                                    t_cost=params['t_cost'],
                                    label=params['label'],
                                    init_holdings=params['init_holdings'],
                                    window_size=params['window_size'])

    my_portfolio.run(min_return=min_return, max_risk=max_risk, num=num)

if __name__ == '__main__':
    with open('config.yml', 'r') as yml:
        cfg = yaml.safe_load(yml)
        locals().update(cfg)
    print(locals())
    t1 = time.time()
    main(**locals())
    t2 = time.time()
    print(f'Total time: {t2-t1}')
    
