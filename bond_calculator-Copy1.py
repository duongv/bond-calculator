import pandas as pd
import edhec_risk_kit as erk
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import collections
import math
import ipywidgets as widgets
import statsmodels.api as sm
import numpy_financial as npf
from IPython.display import display
import re
import colorama
from colorama import Fore, Style
import yfinance as yf
from IPython.display import display, HTML
import plotly.express as px
import plotly.graph_objects as go
from yahoofinancials import YahooFinancials
import yahoo_fin.stock_info as si
import string

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

def summary_stats(r):
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
        "Max Drawdown" : dd })
    f = pd.concat([m,maxdrawdown_date(r)],axis=1) #combine stats with max drawdown date
    # formatting the dataframe
    for x in ['Annualized Return','Annualized Vol','Cornish-Fisher Var 5%',
              'Expected Shortfall','Max Drawdown','Best Year','Worst year']:
        f[x] = f[x].astype(float).map(lambda n: '{:.2%}'.format(n))
    for x in ['Skewness','Kurtosis','Sharpe Ratio']:
        f[x] = f[x].astype(float).map(lambda n: '{:.2f}'.format(n))
    return f 

def interactive_maucaulay(maturity,principal,couponrate,coupon_per_year,ytm):
    """
    computing Maucaulay duration with interaction
    """
    #zero coupon bond
    if coupon_per_year == 0:
        duration = maturity
        modified_duration = maturity 
    
    # coupon bond 
    else: 
        cash_flows =   erk.bond_cash_flows(maturity=maturity,principal=principal,coupon_rate=couponrate,coupons_per_year=coupon_per_year)
        cash_flows.index = cash_flows.index/coupon_per_year
        duration = erk.macaulay_duration(cash_flows,ytm/coupon_per_year)
        modified_duration = duration / (1 + ytm/coupon_per_year)
        modified_duration = modified_duration.round(2)
    
    str1 = Fore.BLUE + "\n" + "The Macaulay duration is : " + str(round(duration,2))
    str2 = "\n" + "The modified Macaulay duration is: " + str(modified_duration)
    
    # setting up the GUI to implement "result" button
    button = widgets.Button(description="Result")
    output = widgets.Output()

    display(button, output)

    def on_button_clicked(b):
        with output:
            
            print (str1)
            print (str2)

    button.on_click(on_button_clicked)
    

def bond_duration_calculator():
    "Calculating Macaulay duration and Modified Duration with interaction"
    style = {'description_width': 'initial'}
    controls = widgets.interactive(interactive_maucaulay,
                                  maturity=widgets.FloatText(min=1,max=100,step=1,value=4),
                                  principal=widgets.FloatText(min=100,max=10000,step=100,value=1000),
                                  couponrate=(0.01,0.2,0.01),
                                  coupon_per_year=widgets.Dropdown(options=[('Zero coupon',0),('Annually',1),('Semi Annually',2),
                                                                            ('Quarterly',4),('Monthly',12)],
                                                                   value=2,
                                                                   description='# of coupons'),
                                                                   ytm=(0.01,0.2,0.01),
                                                                   
                                  )
    display(controls)

def bond_price(maturity,principal,coupon_rate,coupons_per_year,discount_rate):
    """
    compute bond price
    """
    #zero coupon bond
    if coupon_rate == 0.0 and coupons_per_year != 0:
        str1 = "Please select Zero Coupon as # of coupons"
    elif coupons_per_year == 0 :
        if coupon_rate != 0.0:
            str1 = 'Please select coupon_rate = 0 '
        else:
            bond_prices = principal / (1 + discount_rate)**(maturity)
            str1 = Fore.BLUE + "Zero coupon bond price is: " + str(round(bond_prices,2))
    # coupons bond
    else :  
        bond_cash_flow = erk.bond_cash_flows(maturity=maturity,principal=principal,coupon_rate=coupon_rate,
                        coupons_per_year=coupons_per_year)
        bond_prices = erk.pv(bond_cash_flow,discount_rate/coupons_per_year).round(2)
    
        str1 = Fore.BLUE + "\n" + "The price of the " + str(maturity) + " year(s) bond is : " + str(bond_prices[0]) 
    
    # setting up the GUI to implement "result" button
    button = widgets.Button(description="Result")
    output = widgets.Output()

    display(button, output)

    def on_button_clicked(b):
        with output:
            print (str1)
            

    button.on_click(on_button_clicked)  

         

def bond_price_calculator():
    """
    calculating bond price with interation
    """
    gcontrols = widgets.interactive(bond_price,
                                  maturity=widgets.FloatText(min=1,max=40,step=1,value=4),
                                  principal=widgets.FloatText(min=100,max=10000,step=100,value=1000),
                                  coupon_rate=(0.00,0.2,0.005),
                                  coupons_per_year=widgets.Dropdown(options=[('Zero coupon',0),('Annually',1),('Semi Annually',2),
                                                                            ('Quarterly',4),('Monthly',12)], 
                                                                    value = 2,
                                                                    description = "# of coupons"),
                                  discount_rate=widgets.FloatText(min=0.0,max=0.5,step=0.01,value=0.05))
    display(gcontrols)
    

def ytm(maturity,par_value,coupon_rate,coupons_per_year,bond_price):
    """
    compute bond yield to maturity
    """
    if coupons_per_year ==0:
        s1='zero coupon'
    elif coupons_per_year == 1:
        s1='annually coupon'
    elif coupons_per_year == 2:
        s1 = 'semi annually coupon'
    else :
        s1 = 'monthly coupon'
    # compute ytm of zero coupon bond 
    if coupons_per_year == 0:
        ytm = (par_value/bond_price)**(1/maturity) -1 
    # constructing cash flows of the coupon bond
    else: 
        cash_flows = erk.bond_cash_flows(maturity=maturity,
                                         principal=par_value,
                                         coupon_rate=coupon_rate,
                                         coupons_per_year=coupons_per_year)
        #adding par value as the 1st element (your investment)
        first_element = [-bond_price]
        first_element[1:] = cash_flows
        ytm = npf.irr(first_element)
    
    #current yield
    current_yield = coupon_rate / bond_price
    
    # concatenating string for ytm and current yield
    str1 = Fore.BLUE + "\n" + "The YTM of " + s1 + " is: " + str(round(ytm,5)) + "\n"
    str2 = "The current yield of " + s1 + " is: " + str(round(current_yield,5))
    
    # setting up the GUI to implement "result" button
    button = widgets.Button(description="Result")
    output = widgets.Output()

    display(button, output)

    def on_button_clicked(b):
        with output:
            print (str1)
            print (str2)

    button.on_click(on_button_clicked) 
    
def bond_ytm_calculator():
    
    gcontrols = widgets.interactive(ytm,
                                      maturity=widgets.FloatText(min=1,max=40,step=1,value=4),
                                      par_value=widgets.FloatText(min=100,max=10000,step=100,value=1000),
                                      coupon_rate=widgets.FloatText(min=0.01,max=0.2,step=0.1,value=0.03),
                                       coupons_per_year=widgets.Dropdown(options=[('Zero coupon',0),('Annually',1),('Semi Annually',2),
                                                                            ('Quarterly',4),('Monthly',12)], 
                                                                    value = 2,
                                                                    description = "# of coupons"),
                                      bond_price=widgets.FloatText(min=0,max=2000,step=1,value=100))
    display(gcontrols)

def convexity(maturity=6,principal=1000,couponrate=0.08,coupons_per_year=2,discount_rate=0.1):
    """
    compute bond convexity
    """
    #zero coupon bond
    if couponrate == 0.0 and coupons_per_year != 0:
        str1 = "Please select Zero Coupon as # of coupons"
    elif coupons_per_year == 0 :
        if couponrate != 0.0:
            str1 = 'Please select coupon_rate = 0 '
        else:
            str1 = (maturity**2 + maturity) / (1 + discount_rate)**2
    
    # coupon bond 
    else:
        #construct cashflow
        flows= erk.bond_cash_flows(maturity=maturity,principal=principal,coupon_rate=couponrate,coupons_per_year=coupons_per_year)

        #discounted cash flows
        discounted_flow = erk.discount(flows.index,discount_rate/coupons_per_year).multiply(flows,axis='rows')
        # the weight of cashflow/ pv of the bond price 
        discounted_flow['t + t^2'] = discounted_flow.index + discounted_flow.index**2
        discounted_flow['(t + t^2) * PV'] = discounted_flow['t + t^2'] * discounted_flow[0]
        # compute convexity. this is the formula for convexity

        convexity = (discounted_flow['(t + t^2) * PV'].sum() /(discounted_flow[0].sum() * (1 + discount_rate/coupons_per_year)**2)) / coupons_per_year**2
    
        str1 = Fore.BLUE + "\n"+ "The bond convexity is : " + str(convexity)
    
    # setting up the GUI to implement "result" button
    button = widgets.Button(description="Result")
    output = widgets.Output()

    display(button, output)

    def on_button_clicked(b):
        with output:
            
            print (str1)
    button.on_click(on_button_clicked)
    
def bond_convexity_calculator():
    
    gcontrols = widgets.interactive(  convexity,
                                      maturity=widgets.FloatText(min=1,max=40,step=1,value=4),
                                      principal=widgets.FloatText(min=100,max=10000,step=100,value=1000),
                                      couponrate=widgets.FloatText(min=0.01,max=0.2,step=0.1,value=0.03),
                                      coupons_per_year=widgets.Dropdown(options=[('Zero coupon',0),('Annually',1),('Semi Annually',2),
                                                                            ('Quarterly',4),('Monthly',12)], 
                                                                    value = 2,
                                                                    description = "# of coupons"),
                                      discount_rate=(0.01,0.2,0.01))
    display(gcontrols)
    
def bond_calculator_helper(x):
    "Combining YTM , duration and bond price into 1 function"
    print('\n')
    
    if x ==1 :
        return bond_duration_calculator()
    elif x == 2 :
        return bond_price_calculator()
    elif x == 3:
        return bond_ytm_calculator()
    elif x ==4 :
        return bond_convexity_calculator()
    else:
        print("This is not an option")

def bond_calculator():
    "Bond calculator with interation"
    control = widgets.interactive(bond_calculator_helper,
                                     x=widgets.Dropdown(options=[('Duration',1),
                                                                 ('Price',2),
                                                                 ('YTM',3),
                                                                 ('Convexity',4)],
                                                                 description='Solve for'),
                                     )
    display(control)
    
def stock_data(x,start_date,end_date):
    """
    Downloads stock prices from yahoo finance for the last 15 years.
    Returns a df
    """
    k = pd.DataFrame(yf.download(tickers=x,period='15y',proxy=None)['Adj Close'])
    
    # logic for mm/dd/yyyy or mm/yyyy
    count = 0
    for i in start_date:
        if i =='-' or i == '/':
            count = count + 1 
    # specified date range 
    m = pd.DataFrame(k[start_date :end_date])
    # get an element right before the specified period for pct_change method
    if count < 2 : 
        element_b =  pd.DataFrame(k.iloc[k.index.get_loc(start_date).start -1 ]).transpose()   
    else:
        element_b =  pd.DataFrame(k.iloc[k.index.get_loc(start_date) -1 ]).transpose() 
    #concat element right before and the specified date range
    new_range = pd.concat([element_b,m])
    if bool(re.search(r",",x)) is False: #checking if there is space in string
        new_range.columns=[x]
    return new_range

def stockprice_to_monthlyrate_1(x,start_date,end_date):
    """
    convert daily stock price to monthly rate of return 
    """
    # daily stock price 
    d = stock_data(x,start_date,end_date)
    rate_d_1 = d.pct_change() #daily rate of return
    rate_d = rate_d_1.dropna()
    rate_m = rate_d.resample('M').apply(erk.compound) #monthly rate of return
    return rate_m

def monthly_heatmap_single(x,start_date,end_date):
    """
    Returns heatmap of monthly return for a single stock.
    Reminder. This function only works for a single stock 
    """  
    rate_m = stockprice_to_monthlyrate_1(x,start_date,end_date) # simple monthly return 
    rate_m.index = rate_m.index.to_period('M')

    rate = rate_m.copy()
    # configure for the size of the heatmap based on how many years it has 
    r = math.ceil(rate_m.shape[0]/12) *3
    l = r *3 +3
    if r < 7:
        f_size =10
    else:
        f_size =20

    rate['month'] = [i.month for i in rate.index]
    rate['year'] = [i.year for i in rate.index]
    rate = rate.groupby(['month','year']).mean()
    rate = rate.unstack(level=0)

    fig, ax = plt.subplots(figsize=(l,r))
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
    title = x + " monthly return"  + ' from '   + start_date + ' to '  + end_date 
    plt.title(title,fontsize = f_size +5)
    plt.show()

def monthly_heatmap_mutiple(stocks='AAPL,NVDA',start_date='2019',end_date='2020'):
    """
    Returns heatmap of monthly return for multiple stocks.
    Reminder. This function only works for 2 or more stocks 
    """
    from collections import Counter
    s = stocks.split(',') # list of stocks 
    rate_m = stockprice_to_monthlyrate_1(stocks,start_date,end_date) # simple monthly return 
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

def balance_income_cash(stocks,k,features):
    """
    helper function for stock_price funtion to display balance sheet, income statement and cashflow
    """
    from termcolor import colored
    if k == 'balance':
        n = [pd.DataFrame((si.get_balance_sheet(i).iloc[:,[0,1]])) for i in stocks] #extract data from yf and store dataframes in a list
        print(colored('\nSELECTED BALANCE SHEET DATA \n',attrs=['bold']))
        print('in millions')
    elif k == 'income':
        n = [pd.DataFrame((si.get_income_statement(i).iloc[:,[0,1]])) for i in stocks] # extract data from yf and store dataframes in a list
        print(colored('\nSELECTED INCOME STATEMENT DATA \n',attrs=['bold']))
        print('in millions')
    elif k == 'cash':
        n = [pd.DataFrame((si.get_cash_flow(i).iloc[:,[0,1]])) for i in stocks] #extract data from yf and store dataframes in a list
        print(colored('\nSELECTED CASH FLOW DATA \n',attrs=['bold']))
        print('in millions')
    r = pd.concat(n,axis=1) # concat df in the list to be 1 big df
    for i in features:    # if metric not in original df, assign it as n/a value
        if i not in r.index:
            r.loc[i,:]=float("NaN")
    r= r.loc[features,:] # only choose metrics from features list
    for i in r.index.to_list():
        r.loc[i,:] = r.loc[i,:].astype(float).map(lambda n: str('{:,.2f}'.format(n/10**6))) # formatting
    r.columns = pd.MultiIndex.from_tuples(zip(np.repeat(stocks, 2), r.columns)) # nested columns
    return r


def stock_price(x='AAPL',start_date='2019',end_date='2020',options=1):
    """
    Extract stock prices and show their summary statistics from yahoo finance
    """
    #helper function to display balancesheet,income statement and cashflow
    if x =='':
        print('Please type in the stock symbols')
    else:
        stocks = list(x.split(',')) 
        #constructing monthly return graph with interaction 
        if options ==1 : 
            d = stock_data(x,start_date,end_date)
            d.index.name = 'Date' 
            stocknames = d.columns.to_list() # getting stock name
            d.reset_index(inplace=True) # use index as column
            d_1 = pd.melt(d,id_vars='Date',value_vars=stocknames) # rearange from wide data to long data
            
            fig = px.area(d_1,x=d_1['Date'],y=d_1["value"],color=d_1['variable'],
                         title = x + ' stock price' + ' from ' + start_date + ' to ' + end_date,
                         labels={'variable': 'Stocks'},
                         color_discrete_map = {x : 'lightsteelblue'},
                         template= 'simple_white')

            fig.update_yaxes(title_text=' ',tickprefix="$",showgrid=False,) # y-axis in dollar sign

            fig.update_xaxes(title_text=' ',showgrid=False,
                             spikemode ='toaxis',spikedash='dot', # enable spike mode
                             spikecolor='#999999',spikesnap='cursor') # change spike color 

            # modified hovertemplate properties. 
            fig.update_traces(mode="lines", 
                              hovertemplate='%{y:$.2f} <extra> </extra>', # show price 
                              hoverlabel=dict(bgcolor="black",font_color='white'))

            fig.update_layout(hovermode='x') # show the date on axis                     
        
        #constructing summary statistics
        elif options ==2: 
            rate_m = stockprice_to_monthlyrate_1(x,start_date,end_date) # simple monthly return
            r_summary_stats = summary_stats(rate_m)
            
        #constructing monthly rate of return table
        elif options ==3:
            rate_m_1 = stockprice_to_monthlyrate_1(x,start_date,end_date) # simple monthly return
            rate_m_1.index = rate_m_1.index.to_period('M')  # get rid of the date in mm/dd/yyyy
            for x in rate_m_1.columns.to_list():
                rate_m_1[x] = rate_m_1[x].astype(float).map(lambda n: '{:.2%}'.format(n))  #change float to percentage
       
        #constructing quote price table for stocks               
        elif options ==4:
            l = [si.get_quote_table(s) for s in stocks] # extracting data from yahoo finance
            quote = pd.DataFrame(l).transpose() # dictionary -> df
            quote.columns = stocks
            quote.fillna(value=0,inplace=True)
            for i in ['Volume','Avg. Volume']:
                if i in quote.index:
                    quote.loc[i,:] = quote.loc[i,:].astype(int).map(lambda n: '{:,}'.format(n)) # formatting rows 

        #constructing metrics 
        elif options ==5 : 
            m = [(si.get_stats(s).set_index('Attribute')) for s in stocks] # extracting data from yahoo finance, store them in a list
            metrics= pd.concat(m,axis=1) # a list of dataframes  -> df
            metrics.columns = stocks
            # remove the last digit for the index. ex: abcd1 
            r_i = metrics.index.to_list() # list of index
            for x in range(len(r_i)):
                r_i[x] = r_i[x].rstrip(string.digits)
            metrics.index = r_i
        
        elif options ==6:
            features_balance = ['cash','netReceivables','inventory','totalCurrentAssets','propertyPlantEquipment','goodWill',
            'totalAssets','accountsPayable','shortLongTermDebt','totalCurrentLiabilities',
            'longTermDebt','totalLiab','retainedEarnings','commonStock','treasuryStock','totalStockholderEquity']
             
            balance = balance_income_cash(stocks,'balance',features_balance)
        elif options ==7:
            features_income = ['totalRevenue','costOfRevenue','grossProfit','totalOperatingExpenses','operatingIncome',
            'interestExpense','incomeBeforeTax','incomeTaxExpense','netIncome']
            
            income = balance_income_cash(stocks,'income',features_income)
        elif options ==8:
            features_cash = ['netIncome','depreciation','totalCashFromOperatingActivities','capitalExpenditures',
            'totalCashflowsFromInvestingActivities','issuanceOfStock','repurchaseOfStock','dividendsPaid',
            'totalCashFromFinancingActivities']
            
            cash = balance_income_cash(stocks,'cash',features_cash)
        
        # set up 'showing' button 
        button = widgets.Button(description="Result")
        output = widgets.Output()
        display(button, output)
        """
        show the result button 
        """
        def on_button_clicked(b):
            with output:
                if options ==1: # graph
                    return fig.show()
                elif options ==2: # summary_stats
                    display(r_summary_stats)
                elif options ==3: #monthly return df
                    display(rate_m_1)                    
                elif options ==4: # quote price
                    display(quote)  
                elif options ==5: #metric
                    display(metrics) 
                elif options ==6: #balance sheet                  
                    display(balance)
                elif options ==7: # income statement
                    display(income)
                elif options ==8: #cashflow
                    display(cash)
                
        button.on_click(on_button_clicked)
           
def stockprice_visualization():
    "display stocks visualization, summary statistics with interaction"
    c = widgets.interact(stock_price,x=widgets.Text(value='',description='Tickers',placeholder='AAPL,SPY,etc..'),
                         start_date =widgets.Text(value='2019',description='Start Date'),
                         end_date = widgets.Text(value='2021',description='End Date'),
                         options=widgets.RadioButtons(options=[('Stock Price',1),
                                                               ('Summary Statistics',2),
                                                               ('Monthly returns',3),
                                                               ('Quote Price',4),
                                                               ('Metrics',5),
                                                               ('Balance Sheets',6),
                                                               ('Income Statements',7),
                                                               ('Cash flow statements',8)],
                                                               description=' '))
    display(c)
    
def portfolio(x='AAPL',weight='1',initial=1,start_date='2019',end_date='04-2021',benchmark='',show_stats='No'):
    """
    Implements and shows portfolio visualization
    """
    if x =='' or weight =='':
        raise ValueError('Please type in stock symbol(s)') 
    
    #convert list of type string weights to type float weights 
    w = weight.split(',')
    for i in range(0,len(w)):
        w[i] = float(w[i])

    # total of weights = 1 
    if np.sum(w) != 1 : 
        raise ValueError('Sum of weights has to be 1')
    
    stocks = list(x.split(','))
    if len(w) != len(stocks):  # numbers of stocks = numbers of weights
        raise ValueError('# of stocks and # of weights have to be the same')       
    """
    Set up the portfolio
    """
    rate_m = stockprice_to_monthlyrate_1(x,start_date,end_date) # simple monthly return 
    rate_f = pd.DataFrame(np.sum(np.multiply(w,rate_m),axis=1)) #w1 *r1 + w2*r2 +....+ wn *rn
    rate_f1 = initial * (1 + rate_f).cumprod() # (1+r)(1+r1)...
    rate_f1.iloc[0] = initial # set the first row to initial investment so it looks better on graph
    rate_f.columns = ['Your Portfolio']
    rate_f1.columns =['Your Portfolio']
    currency = "${:,.2f}".format(initial)

    """
    Set up the benchmark if benchmark is not none 
    Display summary stats (optional) + graph 
    """
    if benchmark !='':
        rate_benchmark = stockprice_to_monthlyrate_1(benchmark,start_date,end_date) # monthly return of benchmark
        rate_b1 = initial * (1 + rate_benchmark).cumprod() # cumulative returns 
        rate_b1.iloc[0] = initial  # set first row to initial for graph purposes 
        column_list = rate_b1.columns.to_list() # return column names     
        
        # add 'benchmark' prefix  to the stocks
        for i in range(len(column_list)):
            column_list[i] = 'Benchmark (' + column_list[i] +')'
        rate_b1.columns = column_list
        rate_benchmark.columns = column_list 
        
        combining = rate_f1.merge(rate_b1,left_index=True,right_index=True) # combining benchmark, your portfolio dataframes
        
        #summary statistic table 
        summary_stats_df = rate_f.merge(rate_benchmark,left_index=True,right_index=True) # merge your portfolio with benchmark stocks 
        ending_balance = pd.DataFrame(combining.iloc[-1,:]) # ending balance
        ending_balance.columns = ['Ending Balance'] # change column name to ending balance 
        if show_stats =='Yes':
            s = pd.concat([ending_balance,summary_stats(summary_stats_df)],axis=1) # display key stat + ending balance 
            s.iloc[:,0] = s.iloc[:,0].astype(float).map(lambda n: '${:,.2f}'.format(n)) # format ending balance 
            display(s)
            
        #unpivot data for the purpose of graphing 
        stocknames = combining.columns.to_list()
        combining.reset_index(inplace=True) # use index as column
        combining_1 = pd.melt(combining,id_vars=['index'],value_vars=stocknames) # rearange from wide data to long data for the purpose of graphing
        
        # set up graph with portfolio and benchmark
        var = combining_1['variable'] 
        fig = px.line(combining_1,x=combining_1['index'],y=combining_1['value'], color=combining_1['variable'],
                      title = 'Your investment of ' + currency + ' from ' + start_date + ' to ' + end_date,
                      labels={'variable':' ','index':'Year','value':'Portfolio Balance'}    
                     )
        fig.update_yaxes(tickprefix="$", # the y-axis is in dollars
                         showgrid=True) 
                            
        fig.update_xaxes(showgrid=False
                         ,spikemode ='across',spikedash='dot', # enable spike mode
                         spikecolor='#999999',spikesnap='cursor') # change spike color 

        fig.update_layout(hovermode="x",
                          hoverdistance=1000,  # Distance to show hover label of data point
                          spikedistance=1000) # Distance to show spike     
        fig.update_traces(mode = 'lines',
                          hovertemplate ='<br>%{y:$,.2f} <extra></extra>')

        return fig.show()
    
    """
    show summary statistics and graph when benchmark is none 
    """
    # summary statistics table 
    if show_stats =='Yes':
        s = summary_stats(rate_f)
        s.index = ['Your Portfolio'] # change index name
        ending_balance = pd.DataFrame(rate_f1.iloc[-1,:]) # ending balance
        ending_balance.columns = ['Ending Balance'] # change column name to ending balance 
        s1 = pd.concat([ending_balance,s],axis=1)
        s1.iloc[:,0] = s1.iloc[:,0].astype(float).map(lambda n: '${:,.2f}'.format(n)) # format ending balance 
        print('\n'*2)
        display(s1)
        print('\n'*2)

    # main graph 
    fig = px.line(rate_f1,x=rate_f1.index,y=rate_f1['Your Portfolio'],
                  title = 'Your investment of ' + currency + ' from ' + start_date + ' to ' + end_date,
                 labels={'index':'Year'})
                 
    fig.update_yaxes( tickprefix="$",showgrid=True) # the y-axis is in dollars
                    
    fig.update_xaxes(showgrid=False,
                     spikemode ='toaxis',spikedash='dot', # enable spike mode
                     spikecolor='#999999',spikesnap='cursor') # change spike color
                    
    fig.update_traces(mode = 'lines',
                     hovertemplate ='Your Portfolio: %{y:$,.2f}',
                     hoverlabel=dict(bgcolor="black",font_color='white')) # format portfolio value and date month/year
    
    fig.update_layout(hovermode='x')
    return fig.show()

def show():
    control = widgets.interact(portfolio,x=widgets.Text(value='',description='Ticker(s)',placeholder='AAPL,SPY,etc..'),
                     weight =widgets.Text(value='',description='Weight(s)',placeholder='0.1,0.2,0.3,etc...'),
                     benchmark = widgets.Text(value='',description='Benchmark'),
                     initial = widgets.FloatText(value=10000,desciption='Intitial'),
                     start_date =widgets.Text(value='2019',description='Start Date'),
                     end_date = widgets.Text(value='04-2021',description='End Date'),
                     show_stats=widgets.RadioButtons(options=['No','Yes'],description='Statistics'))
                     
                     
    display(control)
    
def portfolio_optimization(x='AAPL,NVDA',your_weights = '.5,.5',riskfree_rate = 0.03,start_date='12-2019',end_date='2020'):
    """
    Visualizing tangency and global min var portfolio for stocks in SP500
    """
    rate_m = stockprice_to_monthlyrate_1(x,start_date,end_date) # simple monthly return 
    er = erk.annualize_return(rate_m,12) # portfolio annual return
    cov = rate_m.cov()
    
    weights = erk.optimal_weights(40,er,cov)
    rets = [erk.portfolio_return(w,er) for w in weights] # annual return
    m_vols = [erk.portfolio_vol(w,cov) for w in weights] # monthly std
    vols = [] # annual volitility
    for i in m_vols:
        vols.append(i * math.sqrt(12))
    
    frontier = pd.DataFrame({'Returns': rets, 'Vols':vols})
    
    # convert list of type string weights to type float weights
    yours_w = your_weights.split(',')
    for i in range(0,len(yours_w)):
        yours_w[i] = float(yours_w[i])
    
    yours_r = erk.portfolio_return(np.array(yours_w),er)
    yours_v = (erk.portfolio_vol(np.array(yours_w),cov))*math.sqrt(12)  
    """
    constructing sharpe ratio
    """
    # constructing weights to find the est sharpe ratio
    sharpe_w = erk.msr(float(riskfree_rate),er,cov)
    # portfolio return with sharpe ratio
    sharpe_r = (erk.portfolio_return(sharpe_w,er)) 
    # portfolio vol with sharpe ratio
    sharpe_v = (erk.portfolio_vol(sharpe_w,cov))*math.sqrt(12)    
    """
    constructing global minimum variance 
    """
    minvariance_w = erk.global_min_var(cov)
    minvariance_r = erk.portfolio_return(minvariance_w,er)
    minvariance_v = (erk.portfolio_vol(minvariance_w,cov)) * math.sqrt(12) # annual std
     
    # special data points DF
    special = pd.DataFrame({'Vols':[sharpe_v,minvariance_v,yours_v],
                           'Rets':[sharpe_r,minvariance_r,yours_r]})    
    """
    displaying weights for min variance and sharpe portfolio
    """
    stocks = x.split(',')
    stocks
    
    def percent_format(s):
        for x in s.columns.to_list():
            s[x] = s[x].astype(float).map(lambda n: '{:.2%}'.format(n)) 
        return s 
    # weight
    weights = pd.DataFrame({'Stocks':stocks,'Sharpe Weights':sharpe_w, 'Min Variance Weights':minvariance_w}).round(3)
    weights = weights.set_index(weights.columns[0]) # set first column as index
    weights = percent_format(weights)
    display(weights)    
    """
    plot frontier and highlight sharpe ratio in plotly
    """
    k = erk.summary_stats(rate_m)[['Annualized Return','Annualized Vol']]
    
    layout = go.Layout(
        title = 'Efficient Frontier of ' + x,
        xaxis = dict(title='Annual Standard Deviation'),
        yaxis = dict(title = "Expected Return")
    )
    fig = go.Figure(layout = layout)
   
    # frontier simulation
    fig.add_trace(go.Scatter(x=frontier['Vols'],y=frontier['Returns'],
                             mode='markers',
                             hovertemplate = 'Expected Return: %{y:.2%} <extra></extra> <br> Standard Deviation: %{x:.2%}' )) 
                            
    # individual stocks
    fig.add_trace(go.Scatter(x=k['Annualized Vol'],y=k['Annualized Return'],mode='markers + text', 
                             text=k.index,
                             textposition="bottom center",
                             hovertemplate = 'Expected Return: %{y:.2%} <extra></extra> <br> Standard Deviation: %{x:.2%}'))
    fig.add_trace(go.Scatter(x=special['Vols'],y=special['Rets'],mode='markers',
                             hovertemplate = 'Expected Return: %{y:.2%} <extra></extra> <br> Standard Deviation: %{x:.2%}'))
                            
    
    fig.add_annotation(x=sharpe_v,y=sharpe_r,text=" Max Sharpe <br> portfolio",arrowhead=3) # show sharpe ratio text
    fig.add_annotation(x=minvariance_v,y=minvariance_r,text="Min variance <br>portfolio",arrowhead=3,ax='400') # show min var text
    fig.add_annotation(x=yours_v,y=yours_r,text='Provided <br> portfolio',arrowhead=3,ax='250',ay='-100') # show your portfolio
    fig.update_layout(showlegend=False)
    fig.show()
    
def optimization_interaction():
    """
    Portfolio optimization with interaction
    """
    control = widgets.interact(portfolio_optimization,
                           x=widgets.Text(value='AAPL,AMZN',description='Ticker(s)',placeholder='AAPL,SPY,etc..'),
                           your_weights =widgets.Text(value='0.5,0.5',description='Weight(s)',placeholder='0.1,0.2,0.3,etc...'),
                           riskfree_rate = widgets.Text(value='0.03',description='InterestRate'),
                           start_date =widgets.Text(value='2019',description='Start Date'),
                           end_date = widgets.Text(value='2020',description='End Date'))
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
    k = widgets.interact(m,o=widgets.ToggleButtons(options=[('Stock Info',1),
                                                        ('Portfolio Simulation',2),
                                                        ('Portfolio Optimization',3)],
                                                        description=' '))