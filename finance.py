import pandas as pd
import edhec_risk_kit as erk
import numpy as np
import matplotlib.pyplot as plt
import math
import ipywidgets as widgets
from IPython.display import display
import re
import colorama
from colorama import Fore, Style
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import yahoo_fin.stock_info as si
import string
from termcolor import colored
import seaborn as sn 
import sys
from datetime import date
from datetime import timedelta
import datetime


df = pd.read_csv('stocks.csv') # list of stock tickers

def maxdrawdown_date(m):
    """
    Returns maxdrawdown date and recovery date 
    from max drawdown as dataframe.  
    """
    wealth_index = (1 + m).cumprod()
    previous_peak = wealth_index.cummax()
    drawdown = (wealth_index - previous_peak)/previous_peak
    l = []
    # The goal here is to find the recovery date when max drawdown happened 
    # the logic is to look for drawdown = 0 and its index > maxdrawdown index. 
    for i in m.columns.to_list():
        x = drawdown[[i]][(drawdown[i] == 0) & (drawdown[i].index > drawdown[i].idxmin())]
        if x.empty:  # stock never recovers from its max drawdown.
            l.append('N/A')
        else:
            l.append(x.index[0])
    date = drawdown.idxmin().tolist()
    date_happened =[date[i].strftime('%m/%Y') for i in range(len(date))]
    for i in range(len(l)):
        if not isinstance(l[i],str):
            l[i] = l[i].strftime('%m/%Y')
    return pd.DataFrame({'Max Drawndown Happened': date_happened,'Recovery from Max Drawdown':l},index=m.columns.to_list())

def best_worst_year(m,options='best'):
    """
    Computes the best and worst annual returns from monthly returns
    """
    rate_y = m.resample('Y').apply(erk.compound)
    if options == 'best':
        return rate_y.max()
    if options == 'worst':
        return rate_y.min()
    
def sharpe_ratio(r):
    """
    return Sharpe ratio from monthly returns. 
    Risk free rate is extracted from Fama - French csv file
    This only works for monthly data
    """
    #read data from french-fama to extract monthly rf
    rf = pd.DataFrame(pd.read_csv('F-F_Research_Data_Factors.CSV',header=0,index_col=0,parse_dates=True,na_values=-99.99)['RF']/100)
    
    # format index to datetime
    rf.index = pd.to_datetime(rf.index,format="%Y%m").to_period('M') 
    r.index = pd.to_datetime(r.index,format="%Y%m").to_period('M')
    
    f = pd.concat([rf[r.index[0]:r.index[-1]],r],axis=1) # combine rf and r 
    f['excess_r'] = f.iloc[:,1] - f.iloc[:,0] # excess return
    ann_vol = erk.annual_vol(r,12) # annual std from montly return
    ann_excess_ret = erk.annualize_return(f['excess_r'],12) # annual excess return 
    return ann_excess_ret / ann_vol

def sortino_ratio(r):
    """
    The function takes monthly returns to compute sortino ratio 
    Sortino ratio: (r - rf) / semi deviation aka std when return < 0
    Presume rf = 0.02
    """
    # convert the annual riskfree rate fro per period
    rf_per_period = (1 + 0.02)**(1/12)-1
    # calculate excess return
    excess_return = r - rf_per_period
    #sortino ratio 
    return erk.annualize_return(excess_return,12) / erk.annual_vol(r[r<0],12)

def summary_stats(r):
    """
    Compute summary statistics based on monthly returns. 
    """
    skew = r.aggregate(erk.skewness)
    kurt = r.aggregate(erk.kurtosis)
    annual_r = r.aggregate(erk.annualize_return,periods_per_year=12)
    annual_v = r.aggregate(erk.annual_vol,periods_per_year=12)
    dd = r.aggregate(lambda r: erk.drawdown(r).Drawdown.min())
    expected_shortfall = r.aggregate(erk.cvar_historic)
    cf_var5 = r.aggregate(erk.var_gaussian,modified=True)
    sharpe = r.aggregate(sharpe_ratio)
    best_year = r.aggregate(best_worst_year,options='best')
    worst_year = r.aggregate(best_worst_year,options='worst')
    sortino = r.aggregate(sortino_ratio)
    m = pd.DataFrame({
        "Annualized Return" : annual_r,
        "Annualized Vol" : annual_v,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher Var 5%": cf_var5,
        "Expected Shortfall" : expected_shortfall,
        "Sharpe Ratio": sharpe,
        "Best Year": best_year,
        "Worst year": worst_year,
        "Max Drawdown" : dd,
        "Sortino ratio":sortino})
    f = pd.concat([m,maxdrawdown_date(r)],axis=1) #combine stats with max drawdown date
    # formatting the dataframe
    for x in ['Annualized Return','Annualized Vol','Cornish-Fisher Var 5%',
              'Expected Shortfall','Max Drawdown','Best Year','Worst year']:
        f[x] = f[x].astype(float).map(lambda n: '{:.2%}'.format(n))
    for x in ['Skewness','Kurtosis','Sharpe Ratio']:
        f[x] = f[x].astype(float).map(lambda n: '{:.2f}'.format(n))
    return f 

def date_chosen(period='1Y',start_date_para=None,end_date_para=None):
    """
    helper function for 'STOCK DATA 2 '
    This function defines the date range from the period chosen by user. 
    It returns a string
    """
    from datetime import datetime as dt
    end_date = date.today()
    if period == '1D':
        start_date = end_date - timedelta(days=1)
    elif period == '1W':
        start_date = end_date - timedelta(days=7)
    elif period == '1M':
        start_date = end_date - timedelta(days=30)
    elif period == 'YTD':
        start_date = end_date.strftime('%Y') + '-01-02'
        start_date = dt.strptime(start_date, '%Y-%m-%d')
    elif period == '6M':
        start_date = end_date - timedelta(days=180)
    elif period == '1Y':
        start_date = end_date - timedelta(days=365)
    elif period == '5Y':
        start_date = end_date - timedelta(days=365*5)
    elif period == 'Max':
        start_date = end_date - timedelta(days=365*15)
    elif period == 'Manually':
        if start_date_para is None or end_date_para is None:
            raise ValueError('Please choose the period from Start Date and End Date boxes')
        else:
            start_date = start_date_para
            end_date = end_date_para
    return '(' + start_date.strftime("%m-%d-%Y") + ' to ' + end_date.strftime("%m-%d-%Y") + ')'

def stock_data_2(x='AAPL',period='1Y',start_date_para=None,end_date_para=None):
    """
    Download stock prices from yahoo finance for the last 15 years.
    User could use period like 1Y,5Y or they could use specific date range. 
    Return stock prices df, monthly rate of returns df. 
    """    
    list_of_stocks = x.split(',')  # split stocks by comma 
    holding_list_df = []  # empty list to hold df 
    for i in list_of_stocks:
        if i in df['Company Name'].tolist(): # if user types in company full name instead of its ticker, convert it to ticker. 
            i = df.loc[df['Company Name'] == i].values[0][0]
        holding_list_df.append(pd.DataFrame(yf.download(tickers=i,period='15y',proxy=None)['Adj Close'])) # download stock prices  one by one 
    for i in range(len(list_of_stocks)):
        holding_list_df[i].rename(columns={'Adj Close': list_of_stocks[i]},inplace=True)  # rename df columns (stock names)
    k = pd.concat(holding_list_df,axis=1) 
    
    end_date = datetime.date.today()
    # if today is not in dataset,for example, weekend or holiday, use the 1st business date prior to today
    while not pd.Timestamp(end_date.strftime("%Y-%m-%d")) in k.index: 
        end_date = end_date - timedelta(days=1)
    # specified date range 
    if period == '1D':
        start_date = end_date - timedelta(days=1)
    elif period == '1W':
        start_date = end_date - timedelta(days=7)
    elif period == '1M':
        start_date = end_date - timedelta(days=30)
    elif period == 'YTD':
        start_date = end_date.strftime('%Y') + '-01-02'
    elif period == '6M':
        start_date = end_date - timedelta(days=180)
    elif period == '1Y':
        start_date = end_date - timedelta(days=365)
    elif period == '5Y':
        start_date = end_date - timedelta(days=365*5)
    elif period == 'Max':
        start_date = k.index[0]
    elif period =='Manually': # user uses period instead of specific start date or end date
        if start_date_para is None or end_date_para is None:
            raise ValueError('Please choose the period from Start Date and End Date boxes')
        else:
            start_date = start_date_para
            end_date = end_date_para
    m = k[start_date:end_date]
    #logic for geting 1 day before the specicied start_date - used in the return formula in percentage
    extra_one_day = k.loc[:start_date].tail(2).iloc[[0]]
    #combine extra one day df and specified date range df
    f = pd.concat([extra_one_day,m])
    
    rate_d_1 = f.pct_change() #daily rate of return
    rate_d = rate_d_1.dropna() # drop the first row -n/a
    
    #solution to fix weird bugs when you convert daily returns -> monthly if you have duplicate tickers
    l = [i for i in range(len(list_of_stocks))]
    rate_d.columns = l
    rate_m = rate_d.resample('M').apply(erk.compound) #monthly rate of return
    rate_m.columns = list_of_stocks
    rate_d.columns = list_of_stocks
    return f,rate_m,rate_d

def monthly_heatmap_single(rate_m):
    """
    Returns the heatmap of monthly return for a single stock.
    Reminder : This function only works for a single stock 
    """  
    rate = rate_m.copy()
    # configure for the size of the heatmap based on how much data it has 
    if rate_m.shape[0] <= 12:
        length = 5
        width = 25
        f_size = 8
    elif rate_m.shape[0] <= 24 and rate_m.shape[0] >12:
        length = math.ceil(rate_m.shape[0]/12) *3
        width = length *4 +3
        f_size = 15
    elif rate_m.shape[0] > 24 and rate_m.shape[0] <= 120:
        length = math.ceil(rate_m.shape[0]/12) *3
        width = length *3
        f_size = 19
    else: 
        length = math.ceil(rate_m.shape[0]/12) *3
        width = length *3 +3
        f_size = 30
    rate['month'] = [i.month for i in rate.index]
    rate['year'] = [i.year for i in rate.index]
    rate = rate.groupby(['month','year']).mean()
    rate = rate.unstack(level=0)

    fig, ax = plt.subplots(figsize=(width,length))
    sn.heatmap(rate,annot=True,square=True,cmap='RdYlGn',fmt=".2f",annot_kws={"fontsize":f_size},linewidths=.5)

    # xticks
    ax.xaxis.tick_bottom()
    xticks_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun','Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.xticks(np.arange(12) + .5,labels=xticks_labels,fontsize=f_size) # replace 1,2,3 .. with jan, feb, march ....

    # ysticks
    plt.yticks(rotation=0,fontsize=f_size) # rotate yticks 
    # axis labels
    plt.xlabel('')
    plt.ylabel('')
    full_name = df.loc[df.Symbol ==rate_m.columns[0]].values[0][1]  # get the company 's name from CSV file 
    title = '\n' + full_name + ' Monthly Returns ('+ rate_m.index[0].strftime('%m/%Y') + ' - ' + rate_m.index[-1].strftime('%m/%Y') + ')' +'\n' 
    plt.title(title,fontsize = f_size +5,fontweight="bold")
    plt.show()
    
def monthly_heatmap_mutiple(stocks='AAPL,NVDA',start_date='2019',end_date='2020'):
    """
    Returns heatmap of monthly return for multiple stocks.
    Reminder. This function only works for 2 or more stocks 
    """
    from collections import Counter
    s = stocks.split(',') # list of stocks 
    rate_m = stockprice_to_rate(stocks,start_date,end_date)[0] # simple monthly return 
    rate_m.index = rate_m.index.to_period('M')
    rate = rate_m.copy()
    
    # configure of the heatmap based on number of years and number of stocks 
    r = (math.ceil(rate_m.shape[0]/12) *3) * math.ceil(len(s)/2)
    l = (r *3 +3) *  math.ceil(len(s)/2)
    #font size 
    if r < 7:
        f_size =10
    else:
        f_size =20
    # modify origininal dataframe so it could be used for heatmap
    month = [i.month for i in rate.index]  # extracting month from the date
    year = [i.year for i in rate.index] # extracting year from the date 
    rate['month'] = month
    rate['year'] = year 
    rate = rate.groupby(['month','year']).mean()
    rate = rate.unstack(level=1)

    #helper method to select columns in multi column system in a df
    years = len(set(year)) # total number of years 
    l1 = list(np.arange(len(s)*years)) # l1 is a list of 1,2,3,4,5,6 ........
    l2 = [l1[i:i+years] for i in range(0,len(l1),years)] # l2 is a list of (1,2), (3,4), (5,6)

    fig, ax = plt.subplots(len(s),figsize=(l,r)) #len(s) is how many subplots we have 

    for i in range(len(s)):
        #heat map 
        sn.heatmap(rate.iloc[:,l2[i]].transpose(),annot=True,square=True,cmap='RdYlGn',fmt=".2f",annot_kws={"fontsize":f_size},linewidths=.5,ax=ax[i])
        ax[i].set_title(s[::-1][i],fontsize=f_size) # title. s [::1] is method to reverse a list   

        #xticks
        xticks_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun','Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax[i].set_xticklabels(xticks_labels,fontsize=f_size) # show text . get_xticklabels function will return the defauft tick labels 

        #yticks
        ax[i].set_yticklabels(set(year),rotation=0,fontsize=f_size) # display year label and rotate it  . set(year) returns unique elements in list of year 

        #labels
        ax[i].set_xlabel('')
        ax[i].set_ylabel('')

    plt.tight_layout() # more space between subplots

def balance_income_cash(stock,k,features):
    """
    helper function for stock_price funtion to display balance sheet, income statement and cashflow
    """
    #if stock in df['Company Name'].tolist(): #if user uses company name instead of ticker, convert company name to ticker
    #    stock = df.loc[df['Company Name'] == stock].values[0][0]
    stock = df.loc[df.Symbol_CompanyName == stock].values[0][0]
    if k == 'balance':
        r = pd.DataFrame((si.get_balance_sheet(stock))) #extract data from yf and store it in df 
    elif k == 'income':
        r = pd.DataFrame((si.get_income_statement(stock))) #extract data from yf and store it in df 
    elif k == 'cash':
        r = pd.DataFrame((si.get_cash_flow(stock))) #extract data from yf and store it in df 
    for i in features:    # if metric not in original df, assign it as n/a value
        if i not in r.index:
            r.loc[i,:]=float("NaN")
    r= r.loc[features,:] # only choose metrics from features list
    for i in r.index.to_list():
        r.loc[i,:] = r.loc[i,:].astype(float).map(lambda n: str('{:,.2f}'.format(n/10**6))) # formatting
    return r

def show_balance_income_cash(stock,k):
    """
    Display either balancesheet, income statement
    """
    features_balance = ['cash','netReceivables','inventory','totalCurrentAssets','propertyPlantEquipment','goodWill',
            'totalAssets','accountsPayable','shortLongTermDebt','totalCurrentLiabilities',
            'longTermDebt','totalLiab','retainedEarnings','commonStock','treasuryStock','totalStockholderEquity']
    features_income = ['totalRevenue','costOfRevenue','grossProfit','totalOperatingExpenses','operatingIncome',
            'interestExpense','incomeBeforeTax','incomeTaxExpense','netIncome']
    features_cash = ['netIncome','depreciation','totalCashFromOperatingActivities','capitalExpenditures',
            'totalCashflowsFromInvestingActivities','issuanceOfStock','repurchaseOfStock','dividendsPaid',
            'totalCashFromFinancingActivities']
    
    # set up 'showing' button 
    button = widgets.Button(description="Result")
    output = widgets.Output()
    display(button, output)
    
    #show the result button 
    def on_button_clicked(b):
        with output:
            if k == 'balance': # graph
                print(colored('\nSELECTED BALANCE SHEET DATA \n',attrs=['bold']))
                print('in millions')
                display(balance_income_cash(stock,k,features_balance))
            elif k == 'income': #monthly return df
                print(colored('\nSELECTED INCOME STATEMENT DATA \n',attrs=['bold']))
                print('in millions')
                display(balance_income_cash(stock,k,features_income))                   
            elif k == 'cash': # quote price
                print(colored('\nSELECTED CASH FLOW DATA \n',attrs=['bold']))
                print('in millions')
                display(balance_income_cash(stock,k,features_cash))  
    button.on_click(on_button_clicked)

def interactive_balance_income_cash():
    c = widgets.interact(show_balance_income_cash,stock=widgets.Combobox(description='Company',placeholder='AAPL',options=df.Symbol_CompanyName.tolist()),
                         k=widgets.Dropdown(options=[('Balance Sheet','balance'),('Income Statement','income'),('Cashflow Statement','cash')],description='show'))
    
    display(c)
    
def stock_price(stock='AAPL',period='1Y',start_date=None,end_date=None,option=1):
    """
    Extract stock prices and show their summary statistics or heatmap from yahoo finance
    """
    button = widgets.Button(description="Result")
    output = widgets.Output()
    display(button, output)
        
    #setting the 'Result' button
    def on_button_clicked(b):
        with output:
            x = df.loc[df.Symbol_CompanyName == stock].values[0][0] #convert full name to stock ticker 
            d = stock_data_2(x,period,start_date,end_date)[0] # fetch data from yf
            d.rename(columns={'Adj Close': x},inplace=True)
            rate_d_1 = d.pct_change() #daily rate of return
            rate_d = rate_d_1.dropna()
            rate_m = rate_d.resample('M').apply(erk.compound) #monthly rate of return

            d.index.name = 'Date'
            d.reset_index(inplace=True) # use index as column

            d_1 = pd.melt(d[1:],id_vars='Date',value_vars=x) # rearange from wide data to long data
            fig = px.line(d_1,x=d_1['Date'],y=d_1["value"],labels={'variable': 'Stocks'},template= 'simple_white')

            fig.update_yaxes(title_text=' ',tickprefix="$",showgrid=False,) # y-axis in dollar sign

            fig.update_xaxes(title_text=' ',showgrid=False,
                             spikemode ='toaxis',spikedash='dot', # enable spike mode
                             spikecolor='#999999',spikesnap='cursor') # change spike color 

            # modified hovertemplate properties.
            if d.iloc[1,1] > d.iloc[-1,1]:  # red line if the return is negative 
                line_c = 'red'
            else:
                line_c = 'chartreuse'    #green if the return is positive
            fig.update_traces(mode="lines", 
                              hovertemplate='%{y:$,.2f} <extra> </extra>', # show price 
                              line_color = line_c,
                              hoverlabel=dict(bgcolor="black",font_color='white'))
           
            if period == 'Manually':
                date_range_title = ' (' + d.Date[1].strftime("%m/%Y") + '-' + d['Date'].iloc[-1].strftime("%m/%Y") + ')'
            else:
                date_range_title = ' since ' + d.Date[1].strftime("%m/%Y")
                    
            current_stock_price = d.iloc[-1,1] # current stock price
            current_percent_change = d.iloc[-1,1]/d.iloc[1,1] -1
            current_price_change = d.iloc[-1,1] - d.iloc[1,1]
            
            if current_price_change < 0:
                current_percent_change = current_percent_change * -1 # take the minus away
                title_text = stock + '<br>' + "{:,.2f} USD".format(current_stock_price) + '</br>' +'{:,.2f}'.format(current_price_change)+'(' + "{:,.2%}".format(current_percent_change)  + ') ↓' + date_range_title
            else:
                title_text= stock + '<br>' + "{:,.2f} USD".format(current_stock_price) + '</br>' + '+' +'{:,.2f}'.format(current_price_change)+'(' + "{:,.2%}".format(current_percent_change)  + ') ↑' + date_range_title

            #constructing monthly return graph with interaction 
            fig.update_layout(hovermode='x', # show the date on axis 
                              title={'text': title_text , 
                                    'y':.9,
                                    'x':0.1,
                                    'xanchor': 'left',    
                                    'yanchor': 'top'}, 
                                    font=dict(family="Arial",size=9),
                                    barmode='stack',
                                    legend=dict(x=0.4, y=-0.3),
                                    legend_orientation="h") 

            if option ==1:
                fig.show() # graph
                if d.shape[0] > 250: # only show statistics if data is MORE than 1 year - 250 days trading in 1 year 
                    display(summary_stats(rate_m)) # summary statistics 
            elif option ==2: #monthly return df
                monthly_heatmap_single(rate_m)
    button.on_click(on_button_clicked) 
        
def stockprice_visualization():
    "display stocks visualization, summary statistics with interaction"
    c = widgets.interact(stock_price,stock=widgets.Combobox(description='Company',placeholder='AAPL',options=df.Symbol_CompanyName.tolist()),
                    period = widgets.Dropdown(options=[('1 day','1D'),
                                                        ('1 week','1W'),
                                                        ('1 month','1M'),
                                                        ('6 months','6M'), 
                                                        ('Year to Date','YTD'),
                                                        ('1 year','1Y'),
                                                        ('5 years','5Y'),
                                                        ('Max','Max'),
                                                        ('Manually choose the period','Manually')],description='Period',diabled=False,value='1Y'),
                     start_date = widgets.DatePicker(description='Start Date'),
                     end_date = widgets.DatePicker(description = 'End Date'),
                     option=widgets.RadioButtons(options=[('Historical Stock Price',1),('Heat Map Monthly Returns',2)],description='show'))
                     
    display(c)
    
def weight_return_portfolio(returns,weights):
    """
    Compute the return of a portfolio daily or monthly with a number of different securities and weights"
    The functions takes a df of returns and a list/array of weights. 
    It returns a df
    """
    if returns.shape[1] != len(weights):  
        raise ValueError('# of stocks and # of weights have to be the same')
    
    # total of weights = 1. it is allowed to have value slightly smaller than 1 in case of odd numbers of stocks
    if np.sum(weights) < 0.9 : 
        raise ValueError('Sum of weights does not add up to 1')
    if np.sum(weights) > 1.01 :
        raise ValueError('Sum of weights is greater than 1')
    
    return pd.DataFrame(np.sum(np.multiply(weights,returns),axis=1))
    
def portfolio(x='AAPL',weight='1',initial=10000,period='1Y',start_date=None,end_date=None,benchmark='',freq='Daily'):
    """
    Implement portfolio visualization and display statistics
    """
    #setting the 'Result' button
    button = widgets.Button(description="Result")
    output = widgets.Output()
    display(button, output)
    
    def on_button_clicked(b):
        with output:
            #convert list of type string weights to type float weights 
            w = weight.split(',')
            for i in range(0,len(w)):
                w[i] = float(w[i])

            x_benchmark = x + ("," + benchmark if benchmark else "") # combine stocks in portfolio and benchmark stocks
            # set up the portfolio   
            if freq == 'Daily':   
                rate_d =stock_data_2(x_benchmark,period=period,start_date_para=start_date,end_date_para=end_date)[2] # daily return
                rate_benchmark = rate_d.iloc[:,len(w):]  # cumulative returns for benchmark 
                rate_portfolio = weight_return_portfolio(rate_d.iloc[:,:len(x.split(','))],w)   #w1 * r1 + w2 *r2 + .... + wn*rn
                rate_portfolio.columns = ['Your Portfolio'] # rename column
                s = summary_stats(rate_portfolio.resample('M').apply(erk.compound)) #summary statistics from daily to monthly 
            else:
                rate_m = stock_data_2(x_benchmark,period=period,start_date_para=start_date,end_date_para=end_date)[1]#monthly return 
                rate_portfolio = weight_return_portfolio(rate_m.iloc[:,:len(x.split(','))],w)   #w1 * r1 + w2 *r2 + .... + wn*rn
                rate_benchmark = rate_m.iloc[:,len(w):]  # cumulative returns for benchmark
                rate_portfolio.columns = ['Your Portfolio'] # rename column
                s = summary_stats(rate_portfolio) #monthly summary stats
                
            # if user types in benchmark
            if benchmark !='':
                #ticker -> full name 
                column_list = rate_benchmark.columns.to_list()
                c = []
                for i in rate_benchmark.columns.to_list():
                    if i in list(df.Symbol): 
                        c.append(df.loc[df.Symbol ==i].values[0][1])  # get the company 's name from CSV file 
                    else:
                        c.append(i)
                rate_benchmark.columns = c
                rate_portfolio = rate_portfolio.merge(rate_benchmark,left_index=True,right_index=True) #combine original rate porfolio with rate benchmark 

            portfolio = initial * (1 + rate_portfolio).cumprod() # (1+r)(1+r1)...
            portfolio.iloc[0] = initial # set the first row to initial investment so it looks better on graph
            ending_balance = pd.DataFrame(portfolio.iloc[-1,:]) # ending balance
            ending_balance.columns = ['Ending Balance'] 
            
            # set up graph with portfolio and benchmark
            stocknames = portfolio.columns.to_list()
            portfolio.reset_index(inplace=True) # use index as column
            portfolio = pd.melt(portfolio,id_vars=['Date'],value_vars=stocknames) # rearange from wide data to long data for the purpose of graphing
            var = portfolio['variable'] 
            currency = "${:,.2f}".format(initial) # format currency for graph title

            fig = px.line(portfolio,x=portfolio['Date'],y=portfolio['value'], color=portfolio['variable'],
                              title = 'Your investment of ' + currency + date_chosen(period,start_date,end_date),
                              labels={'variable':' ','index':'Year','value':'Portfolio Balance'})    

            fig.update_yaxes(tickprefix="$", # the y-axis is in dollars
                             showgrid=True) 

            fig.update_xaxes(showgrid=False,
                             spikemode ='across',spikedash='dot', # enable spike mode
                             spikecolor='#999999',spikesnap='cursor') # change spike color 

            fig.update_layout(hovermode="x",
                              hoverdistance=1000,  # Distance to show hover label of data point
                              spikedistance=1000) # Distance to show spike     
            fig.update_traces(mode = 'lines',hovertemplate ='<br>%{y:$,.2f} <extra></extra>')
            
            fig.show() # displaying graph
            
            # summary statistics
            if rate_portfolio.index[-1] - rate_portfolio.index[0] >= timedelta(days=365) : # only show statistics if data is greater or equal than 1 year
                final_summary_stats = pd.concat([ending_balance,s],axis=1) # combine ending balance with other statistics. 
                final_summary_stats.iloc[:,0] = final_summary_stats.iloc[:,0].astype(float).map(lambda n: '${:,.2f}'.format(n)) # format ending balance 
                display(final_summary_stats) # display summary statistics
                
    button.on_click(on_button_clicked) 
    
def show():
    control = widgets.interact(portfolio,x=widgets.Text(value='',description='Ticker(s)',placeholder='AAPL,SPY,etc..'),
                     weight =widgets.Text(value='',description='Weight(s)',placeholder='0.1,0.2,0.3,etc...'),
                     benchmark = widgets.Text(value='',description='Benchmark'),
                     initial = widgets.FloatText(value=10000,description='Intitial(USD)'),
                     period = widgets.Dropdown(options=[('1 day','1D'),
                                                        ('1 week','1W'),
                                                        ('1 month','1M'),
                                                        ('6 months','6M'),
                                                        ('Year to Date','YTD'),
                                                        ('1 year','1Y'),
                                                        ('5 years','5Y'),
                                                        ('Max','Max'),
                                                        ('Manually choose the period','Manually')],description='Period',diabled=False,value='1Y'),
                     start_date =widgets.DatePicker(description='Start Date'),
                     end_date = widgets.DatePicker(description='End Date'),
                     freq=widgets.RadioButtons(options=['Daily','Monthly'],description='Period'))
    display(control)
    
def portfolio_optimization(x='AAPL,NVDA',your_weights = '0.6,0.4',riskfree_rate = 0.03,period='5Y',start_date=None,end_date=None):
    """
    Visualizing tangency and global min var portfolio for stocks in SP500
    """
    button = widgets.Button(description="Result")
    output = widgets.Output()
    display(button, output)
        
    #setting the 'Result' button
    def on_button_clicked(b):
        with output:
            # convert list of type string weights to type float weights
            yours_w = your_weights.split(',')
            for i in range(0,len(yours_w)):
                yours_w[i] = float(yours_w[i])

            rate_m = stock_data_2(x,period,start_date,end_date)[1] # simple monthly return 
            er = erk.annualize_return(rate_m,12) # portfolio annual return
            cov = rate_m.cov()

            weights_frontier = erk.optimal_weights(120,er,cov)
            rets_f = [erk.portfolio_return(w,er) for w in weights_frontier] # annual return
            m_vols = [erk.portfolio_vol(w,cov) for w in weights_frontier] # monthly std
            vols_f = [] # annual volitility
            for i in m_vols:
                vols_f.append(i * math.sqrt(12))

            frontier = pd.DataFrame({'Returns': rets_f, 'Vols':vols_f})
            yours_r = erk.portfolio_return(np.array(yours_w),er) # given portfolio return
            yours_v = (erk.portfolio_vol(np.array(yours_w),cov))*math.sqrt(12)  #given portfolio volatility
            yours_m = weight_return_portfolio(rate_m,yours_w)

            #constructing sharpe ratio
            sharpe_w = erk.msr(float(riskfree_rate),er,cov) # constructing weights to find the best sharpe ratio
            sharpe_r = (erk.portfolio_return(sharpe_w,er)) # portfolio return with sharpe ratio
            sharpe_v = (erk.portfolio_vol(sharpe_w,cov))*math.sqrt(12)  # portfolio vol with sharpe ratio    
            sharpe_m = weight_return_portfolio(rate_m,sharpe_w) #monthly return of your portfolio with Sharpe weights

            #constructing global minimum variance 
            minvariance_w = erk.global_min_var(cov)
            minvariance_r = erk.portfolio_return(minvariance_w,er)
            minvariance_v = (erk.portfolio_vol(minvariance_w,cov)) * math.sqrt(12) # annual std
            minvariance_m= weight_return_portfolio(rate_m,minvariance_w) #monthly return of your portfolio with Sharpe weights

            # special data points DF
            special = pd.DataFrame({'Vols':[yours_v,sharpe_v,minvariance_v],
                                           'Rets':[yours_r,sharpe_r,minvariance_r],
                                          'Weights':[yours_w,sharpe_w,minvariance_w]})
            # set up hovertemplate details
            n_stocks = len(x.split(',')) #number of stocks 
            title = ['<b>Your Portfolio</b><br>','<b>Sharpe Portfolio</b><br>','<b>Min Variance Porfolio</b><br>'] # title for special portfolios
            weight_words = ['<br>Weight:<br>'] # list of the word 'weight'
            stocks = x.split(',') # list of stock tickers 
            colon = [': '] # list of colon
            weights =  [str(round(x*100,2))+"%" for x in yours_w + sharpe_w.tolist() + minvariance_w.tolist()] # list of individual security weight for special p
            return_words = ['Exptected Return: '] # list of the term 'expected returns'
            vols_words = ['Standard Deviation: '] # list of the term 'Standard Deviation'
            returns =  [str(round(x*100,2))+"%" for x in special['Rets'].tolist()] # list of frontier returns
            vols = [str(round(x*100,2))+"%" for x in special['Vols'].tolist()] # list of frontier std
            space = ['<br>'] # list of <br>
            first_space = ['<br>']
            w_frontier =  [str(round(x*100,2))+"%" for x in np.concatenate(weights_frontier).tolist()] # list of individual security weight for efficient frontier 
            frontier_returns = [str(round(x*100,2))+"%" for x in rets_f] # list of frontier returns
            frontier_vols = [str(round(x*100,2))+"%" for x in vols_f] # list of frontier std. 
            
            def big_zip(title,weight_words,stocks,colon,weights,returns,vols,return_words,vols_words,first_space):
                """
                helper function to set up hovertemplate details for plotly efficient frontier graph. 
                """
                colons = colon * len(stocks) # multiply list of colon by the number of points on efficient frontier
                spaces = space * len(stocks) # multiply list of space by the number of points on  efficient frontier
                a = zip(stocks,colons,weights,spaces) #zip stocks, colons weights and spaces 
                b = [] 
                new_list = []
                final = []
                final_1 =[]
                for i in a:
                    b.append(i[0]+i[1] +i[2] + i[3]) #combine stocks, colons, weights and spaces in a list 
                for i in range(0,len(b),n_stocks):
                    new_list.append(''.join(b[i:i+n_stocks])) #concatenate strings from the list above into big strings and store them in a list. Each string represents one point on efficent frontier graph. 
                for i in range(0,len(title)): # hovertemplate details for each special portfolio
                    final.append(title[i]+ return_words[i] + returns[i] + first_space[i] + vols_words[i]  + vols[i] + weight_words[i] + new_list[i]) 
                for i in range(0,len(return_words)): #hovertemplate details for each frontier portfolio
                    final_1.append(return_words[i] + returns[i] + first_space[i] + vols_words[i]  + vols[i] + weight_words[i] + new_list[i])
                return final,final_1
            
            # 3 special portfolios - provided, min variance and sharpe por
            special_hover = big_zip(title,weight_words*3,stocks*3,colon,weights,returns,vols,return_words*3,vols_words*3,first_space*3)[0]
            # 60 generated frontier portfolio 
            frontier_hover = big_zip(title,weight_words*120,stocks*120,colon,w_frontier,frontier_returns,frontier_vols,return_words*120,vols_words*120,first_space*120)[1]
            
            #plot frontier and highlight sharpe ratio in plotly
            k = erk.summary_stats(rate_m)[['Annualized Return','Annualized Vol']]
            layout = go.Layout(
                title = 'Efficient Frontier of ' + x + date_chosen(period,start_date,end_date),
                xaxis = dict(title='Annual Standard Deviation'),
                yaxis = dict(title = "Expected Return"))

            fig = go.Figure(layout = layout)
            # frontier simulation
            fig.add_trace(go.Scatter(x=frontier['Vols'],y=frontier['Returns'],
                                     mode='markers',
                                     hovertemplate  ='%{text}<extra></extra>',text = frontier_hover)) 

            fig.update_xaxes(showgrid=False,tickformat = ',.0%')
            fig.update_yaxes(showgrid=False,tickformat = ',.0%')
            # individual stocks
            fig.add_trace(go.Scatter(x=k['Annualized Vol'],y=k['Annualized Return'],mode='markers + text', 
                                     text=k.index,
                                     textposition="bottom center",
                                     hovertemplate = 'Expected Return: %{y:.2%} <extra></extra> <br> Standard Deviation: %{x:.2%}'))
            #special portfolios - Your portfolio, sharpe and min variance . 
            fig.add_trace(go.Scatter(x=special['Vols'],y=special['Rets'],mode='markers',textposition='bottom center',hovertemplate  ='%{text}<extra></extra>',text = special_hover))
            fig.update_layout(showlegend=False,width=1200,height=600)
            
            fig.add_annotation(x=sharpe_v,y=sharpe_r,text="<b>Max Sharpe</b>",arrowhead=3) # show sharpe ratio arrow
            fig.add_annotation(x=minvariance_v,y=minvariance_r,text="<b>Min<br>Variance</b>",arrowhead=3,ax='-100',ay='40') # show min var arrow
            fig.add_annotation(x=yours_v,y=yours_r,text='<b>Provided <br> portfolio</b>',arrowhead=3,ax='80',ay='50') # show your portfolio arrow
            
            #summary statistics. 
            con = pd.concat([minvariance_m,sharpe_m,yours_m,rate_m],axis=1)
            con.columns = ['Min Variance Portfolio','Sharpe Portfolio','Your Portfolio'] + x.split(',')

            fig.show() # graph
            display(summary_stats(con))
            
    button.on_click(on_button_clicked)
    
def optimization_interaction():
    """
    Portfolio optimization with interaction
    """
    control = widgets.interact(portfolio_optimization,
                           x=widgets.Text(value='AAPL,AMZN',description='Ticker(s)',placeholder='AAPL,SPY,etc..'),
                           your_weights =widgets.Text(value='0.5,0.5',description='Weight(s)',placeholder='0.1,0.2,0.3,etc...'),
                           period = widgets.Dropdown(options=[('1 day','1D'),
                                                        ('1 week','1W'),
                                                        ('1 month','1M'),
                                                        ('6 months','6M'), 
                                                        ('Year to Date','YTD'),
                                                        ('1 year','1Y'),
                                                        ('5 years','5Y'),
                                                        ('Max','Max'),
                                                        ('Manually choose the period','Manually')],description='Period',diabled=False,value='1Y'),
                           riskfree_rate = widgets.Text(value='0.03',description='InterestRate'),
                           start_date =widgets.DatePicker(description='Start Date'),
                           end_date = widgets.DatePicker(description='End Date'))
    display(control) 
    
def stock_info_optimization():
    """
    combining stock info, portfolio simulation, optimization with interaction
    """
    def m(o=1):
        if o==1:
            return stockprice_visualization()
        elif o==2:
            return show()
        elif o==3:
            return optimization_interaction()
        elif o==4:
            return interactive_balance_income_cash()
    k = widgets.interact(m,o=widgets.ToggleButtons(options=[('Stock Info',1),('Portfolio Simulation',2),
                                                            ('Portfolio Optimization',3),('Accounting Statements',4)],
                                                  description=' '))
    
def monthly_payment(principal,annual_rate,months,extra_monthly_payment):
    """
    Compute monthly payments and other loan information  for amortized loan. 
    """
    if annual_rate <= 0: # check if annual interest rate is greater 0
        print('\n The annual rate has to be greater than 0')
        raise TypeError("Incorrect interest rate")
    m_rate = annual_rate / 12
    p = (principal* m_rate*(1+m_rate)**months) / ( (1+m_rate)**months -1)   # using formula to calculate monthly payment
    p = round(p,2)
    
    def calculation(principal,annual_rate,months,extra_monthly_payment):
        #amortization table
        unpaid_balances = [principal]
        payments = [p]* (months+1)
        interests = [0]
        principal_payments = [0]
        extra_monthly_payments = [extra_monthly_payment]*(months+1)

        for i in range(1,len(payments)):
            interest = m_rate * unpaid_balances[i-1] #calculate monthly interest from principal and monthly interest
            interests.append(interest)

            # loan balance < payment (extra + monthly ) , then payment = loan balance
            if unpaid_balances[i-1] <= (payments[i-1] + extra_monthly_payments[i-1]):   
                if unpaid_balances[i-1] <= payments[i-1]:
                    payments[i] = unpaid_balances[i-1] + interests[i]
                extra_monthly_payments[i] = unpaid_balances[i-1] - payments[i] + interests[i]

            #calculate monthly principal applied from monthly payment
            principal_payment= payments[i] + extra_monthly_payments[i] - interests[i] # payment + extra payment - interest
            principal_payments.append(principal_payment)

            #calcualte unpaid balance 
            unpaid_balance = unpaid_balances[i-1] - principal_payments[i]  # new balance = old balance - principal_applied
            unpaid_balances.append(unpaid_balance)


        #set up data frame of loan armotization 
        df = pd.DataFrame({'Payment':payments,
                           'Extra Monthly Payments': extra_monthly_payments,
                           'Interest':interests,
                           'Principal Amount Applied':principal_payments,
                           'Balance Owed' : unpaid_balances})
        df.iloc[0,0] = 0 # payment for period 0 is set to 0 
        df.iloc[0,1]=0   # extra payment for period 0 is set to 0
     
        return df,payments
    
    df = calculation(principal,annual_rate,months,extra_monthly_payment)[0]    
    amount_savings = round(p * months - np.sum(df.Payment),2)  # the money saved by paying extra monthly

    total_paid = np.sum(df.Payment) + np.sum(df['Extra Monthly Payments']) # total payments 
    total_interest_paid = np.sum(df.Interest) # total interest paid
    interest_saved = np.sum(calculation(principal,annual_rate,months,0)[0].Interest) - np.sum(calculation(principal,annual_rate,months,extra_monthly_payment)[0].Interest)  # interest saved from paying extra monthly payment. 
    number_months_saved = calculation(principal,annual_rate,months,extra_monthly_payment)[1].count(0) # number of loan term in months shorterned by paying extra
    
    df_show = df.copy() # display this pandas frame amortization table 
    for x in df.columns.to_list():
        df_show[x] = df_show[x].map(lambda n: '${:,.2f}'.format(n))  #change float to dollars 
    
    # setting up the GUI to implement "result" button
    button = widgets.Button(description="Result")
    output = widgets.Output()

    display(button, output)

    def on_button_clicked(b):
        with output:
            print('\n' + colored('Amortization Table',attrs=['bold']) + '\n')
            display(df_show)
            print('\n' + 'Monthly payments: ' + colored("${:,.2f}".format(extra_monthly_payment + p),attrs=['bold']))
            print('\n' + 'Totally Paid: ' + colored("${:,.2f}".format(total_paid),attrs=['bold']))
            print('\n' + 'Interest Paid : ' + colored("${:,.2f}".format(total_interest_paid),attrs=['bold']))
    
            if extra_monthly_payment > 0:
                print('\n' + 'You could shorten your loan by ' + colored(number_months_saved,attrs=['bold']) + ' months with extra ' 
                      + colored("${:,.2f}".format(extra_monthly_payment),attrs=['bold']) +' monthly payment')
                print('\n' + 'You could also save ' + colored("${:,.2f}".format(interest_saved),attrs=['bold']) + 
                      ' in interest over the life of the loan')
    button.on_click(on_button_clicked)


def monthly_payment_interactive():
    """
    Interactive loan calculator for monthly payments
    """
    gcontrols = widgets.interactive(monthly_payment,
                               principal=widgets.FloatText(min=0,value=100,description='Principal'),
                               annual_rate = widgets.FloatText(min=0,value=0.1,description='Annual Rate'),
                               months=widgets.IntText(min=0,value=12,description='# of Months'),
                               extra_monthly_payment =widgets.IntText(min=0,value=0,description='Extra Payment'),
                               amortization=widgets.Dropdown(options=[('False',0),('True',1)],description='Schedule'))
    display(gcontrols)




