import pandas as pd
import scipy.stats
import numpy as np
from scipy.optimize import minimize
import ipywidgets as widgets
import math
import statsmodels.api as sm
import yfinance as yf

def yahoo_finance(*a):
    "return daily stock prices from 2010-01-01 to 2020-12-30 "
    data = yf.download(tickers=a,start='2010-01-01',end='2020-12-31',interval='1d',proxy= None)['Close']
    return data


def compound(r):
    "return compound return "
    return np.prod(r + 1) - 1 

def drawdown(return_series: pd.Series):
    """
    Takes a times series of asset returns 
    Computes and returns a dataframe contains:
    the wealth index
    the previous peaks
    percent drawdows
    """
    wealth_index = 1000*(1 + return_series).cumprod()  # cummunitive product 
    previous_peaks = wealth_index.cummax()             # find the max return 
    drawdowns = (wealth_index - previous_peaks)/previous_peaks   # difference between previous peaks vs wealth
    return pd.DataFrame({
        "wealth":wealth_index,
        "Peaks" : previous_peaks,
        "Drawdown" : drawdowns
    })

def get_ffme_returns():
    "Load the Fama French dataset for the returns of the top and bottom deciles by marketcap"
    price_s = pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv",
                      header=0,
                      index_col=0,
                      parse_dates=True,
                      na_values = -99.99)
    # only extract low10 and hi10 column
    returns_4 = price_s[['Lo 10','Hi 10']]
    # divide by 100
    returns_4 = returns_4/100
    #rename
    returns_4.columns = ['SmallCap','LargeCap']
    #change index format
    returns_4.index = pd.to_datetime(returns_4.index,format="%Y%m").to_period('M')
    return returns_4

def get_ind_returns(ew=False):
    """
    returns industry 30 years monthly return 
    """
    if ew:
        return get_ind_file('returns',ew=True)
    else: 
        return get_ind_file('returns',ew=False)

def get_ind_nfirms():
    """
    returns the number of firms in the industry 
    """
    return get_ind_file('nfirms')


def get_ind_size():
    """
    Load and format the Ken French 30 Industry Port Value weighted monthly returns
    """
    return get_ind_file('nzise')

def get_hfi_return():
    "Load and format the EDHEC Hedge Fund Index Returns"
    hfi = pd.read_csv("data/edhec-hedgefundindices.csv",
                      header=0,
                      index_col=0,
                      parse_dates=True,
                      na_values = -99.99)
    # divide by 100 to get the actual number instead of percentage
    hfi = hfi/100    
    #change index format to monthly period
    hfi.index = hfi.index.to_period('M')
    return hfi

def get_fff_return():
    "Load and format Fama-French factors"
    fff = pd.read_csv('data/F-F_Research_Data_Factors_m.csv',
                 header=0,
                 index_col=0,
                 parse_dates=True,
                 )/100
    fff.index = pd.to_datetime(fff.index,format="%Y%m").to_period('M')
    return fff

def summary_stats(r,riskfree_rate=0.03):
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    annual_r = r.aggregate(annualize_return,periods_per_year=12)
    annual_v = r.aggregate(annual_vol,periods_per_year=12)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    expected_shortfall = r.aggregate(cvar_historic)
    cf_var5 = r.aggregate(var_gaussian,modified=True)
    sharpe = r.aggregate(sharpe_ratio,riskfree_rate=riskfree_rate,periods_per_year=12)
    
    return pd.DataFrame({
        "Annualized Return" : annual_r,
        "Annualized Vol" : annual_v,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher Var 5%": cf_var5,
        "Expected Shortfall" : expected_shortfall,
        "Sharpe Ratio": sharpe,
        "Max Drawdown" : dd
    })

def get_ind_file(filetype, ew=False):
    """
    Load and format the Ken French 30 Industry Portfolios files
    """
    known_types = ["returns", "nfirms", "size"]
    if filetype not in known_types:
        raise ValueError(f"filetype must be one of:{','.join(known_types)}")
    if filetype == "returns":
        name = "ew_rets" if ew else "vw_rets"
        divisor = 100
    elif filetype == "nfirms":
        name = "nfirms"
        divisor = 1
    elif filetype == "size":
        name = "size"
        divisor = 1
                         
    ind = pd.read_csv(f"data/ind30_m_{name}.csv", header=0, index_col=0)/divisor
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def skewness(r):
    "computes the skewness of the Dataframe. returns a float or a Series"
    
    demeaned_r = r - r.mean()
    #use the population std, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    """"computes the Kurtosis of the Dataframe
    Returns a float or a Series"""
    
    demeaned_r = r - r.mean()
    #use the population std, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

def isnormal(r,level=0.01):
    """
    Applies the Jarque_bera test to determine if a Series is normal or not 
    Test is applied at 1% level by default
    """
    # unpacked the tuple
    statistics,p_value = scipy.stats.jarque_bera(r)
    return p_value > level

def semideviation(r):
    """
    Returns the semideviation aka negative semideviation of r 
    r must be a Series or Dataframes
    """
    is_negative = r < 0 
    return r[is_negative].std(ddof=0)

def var_historic(r,level=5):
    """
    Returns the historic value at risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number. 
    """
    # check if it is a daframe
    if isinstance(r,pd.DataFrame):
        return r.aggregate(var_historic,level=level)
    
    # series -> return VAR
    elif isinstance(r,pd.Series):
        return -np.percentile(r,level)
    
    # throw an error if it is something else 
    else:
        raise TypeError("expected r to be Series or Dataframe")


    
from scipy.stats import norm
def var_gaussian(r,level=5,modified = False):
    """
    Returns the parametric Gaussian VAR of a Series or Dataframe
    """
    # compute z score assuming it is normally distribution
    z = norm.ppf(level/100)
    if modified:
        #modify the z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z + 
                 (z**2 - 1 )*s/6 +
                 (z**3 -3*z)*(k-3)/24 -
                 (2*z**3 -5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))

def cvar_historic(r,level=5):
    """
    Compute the expected shortfall 
    """
    if isinstance(r,pd.Series):
        is_beyond = r <= -var_historic(r,level=level)
        return - r[is_beyond].mean()
    if isinstance(r,pd.DataFrame):
        return r.aggregate(cvar_historic,level=level)
    else:
        raise TypeError("Expected r to be a Series or Dataframe")
        
def annualize_return(r,periods_per_year):
    """
    Annualizes the vol of a set of returns
    We should infer the periods per year
    """
    compounded_growth = (1+r).prod()  #(1+r)(1+r1)(1+r2)....(1+rn)
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

def annual_vol(r,periods_per_year):
    """
    Annualizes the vol of a set of returns
    we should infer the periods per year 
    """
    return r.std()*(periods_per_year**0.5)


def sharpe_ratio(r,riskfree_rate,periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of return
    """
    # convert the annual riskfree rate fro per period
    rf_per_period = (1 + riskfree_rate)**(1/periods_per_year)-1
    # calculate excess return
    excess_return = r - rf_per_period
    # calculate annual excess return   
    ann_ex_ret = annualize_return(excess_return,periods_per_year)
    # calculate annual volitility
    ann_vol = annual_vol(r,periods_per_year)
    return ann_ex_ret/ann_vol


def portfolio_return(weights,returns):
    """
    Weight -> Returns
    """
    return weights.T @ returns

def portfolio_vol(weights,covmat):
    """
    weights -> vol. return the std of the portfolio
    """
    return (weights.T @ covmat @ weights)**0.5

def plot_ef2(n_points,er,cov):
    """
    Plots 2 securities frontier
    """
    if er.shape[0] != 2 : 
         raise TypeError("expected Portfolio of 2 securities")
    # constructing weight from 0 to 1 
    weights = [np.array([w,1-w]) for w in np.linspace(0,1,n_points)]
    # constructing return of porfolio with w from 0 -> 1 
    rets = [portfolio_return(w,er) for w in weights]
    #constructing volitiltity of port with w from 0 -> 1 
    vols = [portfolio_vol(w,cov) for w in weights]
    
    frontier = pd.DataFrame({"Return":rets,
                            "Vol":vols})
    return frontier.plot.line(x="Vol",y="Return",style='.-')

def optimal_weights(n_points,er,cov):
    """
    list of weights to run the optimizer on to mininmize the vol
    """
    target_rs = np.linspace(er.min(),er.max(),n_points)
    weights = [minimize_vol(target_return,er,cov) for target_return in target_rs]
    return weights
    
def plot_ef(n_points,er,cov,show_cml=False,style='.-',rf=0,show_ew=False,show_gmv=False):
    """
    Plots N securities frontier with option of Sharpe ratio
    """
    # constructing weight from 0 to 1 
    weights = optimal_weights(n_points,er,cov)
    # constructing return of porfolio with w from 0 -> 1 
    rets = [portfolio_return(w,er) for w in weights]
    #constructing volitiltity of port with w from 0 -> 1 
    vols = [portfolio_vol(w,cov) for w in weights]
    
    frontier = pd.DataFrame({"Return":rets,
                            "Vol":vols})
    ax = frontier.plot.line(x="Vol",y="Return",style=style)
    ax.set_xlim(left=0)
    # constructing sharpe ratio portfolio and CMV
    if show_cml:
        # constructing weights to find the best sharpe ratio
        w_msr = msr(rf,er,cov)
        # return of the portfolio with the best sharpe ratio
        r_msr = portfolio_return(w_msr,er)
        # vol of port with the best sharpe ratio
        vol_msr = portfolio_vol(w_msr,cov)

        # Add Captial market line 
        cml_x = [0,vol_msr]
        cml_y = [rf,r_msr]
        # adding sharpe ratio portfolio to the plot
        ax.plot(cml_x,cml_y,color="green",marker="o",linestyle="dashed",markersize=12)
    
    # constructing equally weighted portfolio 
    if show_ew:
        n_2 = er.shape[0]
        weights = np.repeat(1/n_2,n_2)
        r_ew = portfolio_return(weights,er) # return of equally weighted port
        vol_ew = portfolio_vol(weights,cov) # vol of equally weighted port
        
        # add the this portfolio to the plot
        ax.plot(vol_ew,r_ew,color="yellow",marker="o",markersize="14")
       
    
    #  plotting  minimum variance portfolio 
    if show_gmv:
        w_gmv = global_min_var(cov)
        r_gmv =portfolio_return(w_gmv,er)
        vol_gmv = portfolio_vol(w_gmv,cov)
        
        # add this min variance portfolio to the plot
        ax.plot(vol_gmv,r_gmv,color="red",marker= "o",markersize="14")
        
       
    return ax 

def minimize_vol(target_return,er,cov):
    """
    target_ret -> W 
    """
    n= er.shape[0]
    # creating weight equally.
    init_guess = np.repeat(1/n,n)
    # constructing constraint for weight (ex : <1 and > 0)
    bounds = ((0.0,1.0),)*n
    return_is_target = {
        'type': 'eq',
        'args': (er,),
        'fun': lambda weights,er: target_return - portfolio_return(weights,er)
    }
    # weights sum to 1 
    weights_sum_to_1 ={
        'type' : 'eq',
        'fun' : lambda weights : np.sum(weights) - 1 
    }
    results = minimize(portfolio_vol,init_guess,
                       args=(cov,), method="SLSQP",
                       options={'disp': False},
                       constraints=(return_is_target,weights_sum_to_1),
                       bounds=bounds
                       )
    
    return results.x

def global_min_var(cov):
    """
    finding the portfolio with lowest variance
    """
    # number of securities are in the the portfolio
    n_1 = cov.shape[0]
    #creating weight equally
    init_guess_1 = np.repeat(1/n_1,n_1)
    
    #weights are in (0,1)
    bounds_1 = ((0.0,1.0),) * n_1
    # weights sum to 1 
    weights_sum_to_1 = {
        'type':'eq',
        'fun': lambda weights: np.sum(weights) - 1 
    }
    results = minimize(portfolio_vol,init_guess_1,
                      args=(cov,),method="SLSQP",
                      constraints=(weights_sum_to_1),
                      bounds = bounds_1)
    return results.x
                    
    
def msr(riskfree_rate, er,cov):
    """
    riskFree rate + er + cov -> W aka maximise the ne negative of sharpe ratio
    """
    n= er.shape[0]
    # creating weight equally.
    init_guess = np.repeat(1/n,n)
    # constructing constraint for weight (ex : <1 and > 0)
    bounds = ((0.0,1.0),)*n
    # weights sum to 1 
    weights_sum_to_1 ={
        'type' : 'eq',
        'fun' : lambda weights : np.sum(weights) - 1 
    }
    # define the negative sharpe ratio 
    def neg_sharpe_ratio(weights,riskfree_rate,er,cov):
        """
        return the negative of sharpe ratio given weight
        """
        r = portfolio_return(weights,er)
        vol = portfolio_vol(weights,cov)
        return - (r - riskfree_rate)/vol
    # minimize the negative sharpe ratio is the same as maximize ratio
    results = minimize(neg_sharpe_ratio,init_guess,
                       args=(riskfree_rate,er,cov,), method="SLSQP",
                       options={'disp': False},
                       constraints=(weights_sum_to_1),
                       bounds=bounds
                       )
    
    return results.x
    
def run_cppi (risky_r,m,initial_value=1000,floor=0.8,riskfree_rate=0.03,safe_r=None,drawdown=None):
    """
    Run a backtest of CPPI strategy, given a set of returns for the risky asset
    Returns a dictionary containing: asset value history, risk budget history,and Risky Weight History
    """
    if isinstance(risky_r,pd.Series):
        risky_r = pd.DataFrame(risky_r)
        
    if safe_r is None: 
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r[:] = riskfree_rate/12
             
    # getting the dates
    dates = risky_r.index
    # count the number of elements 
    n_steps = len(dates)
    account_value = initial_value
    floor_value = initial_value * floor
    peak = initial_value
    #creating empty dafraframes to store values
    account_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
        
    for step in range(n_steps):
        if drawdown is not None:
            peak =np.maximum(peak,account_value)
            floor_value = peak*(1 - drawdown)
        cushion = (account_value - floor_value) / account_value
        risky_w = m * cushion # compute weights for risky asset
        risky_w = np.minimum(risky_w,1 ) # risky weight < 1 
        risky_w = np.maximum(risky_w,0 ) # risky weight > 0 
        safe_w = 1 - risky_w # compute weights for risk free asset
        risky_alloc = account_value * risky_w
        safe_alloc = account_value* safe_w
        # update the account value for this time step
        account_value = risky_alloc * (1 + risky_r.iloc[step]) + safe_alloc* (1 +safe_r.iloc[step])
        # save the values so i can look at the history and plot it etc
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value
        risky_wealth = initial_value *(1 + risky_r).cumprod()
    values = {
        "Risk Budget" : cushion_history,
        "Risk Allocation " : risky_w_history,
        "Wealth" : account_history,
        "m":m,
        "start": initial_value,
        "floor":floor,
        "risky_r":risky_r,
        "safe_r":safe_r,
        "Risky Wealth": risky_wealth
        }
    return values
        
def gbm(n_years=10,n_scenarios=10,mu=0.07,sigma=0.15,steps_per_year=12,s_0=100.0,prices=True):
    """
    Evolution of a Stock Price using a Geometric Brownian Motion Model
    """
    dt = 1/steps_per_year
    n_steps = int(n_years * steps_per_year)
    rets_plus_1  = np.random.normal(loc=1 + mu*dt,scale=sigma*np.sqrt(dt),size=(n_steps +1 ,n_scenarios))   #loc= mean, scale = std
    rets_plus_1[0] = 1 # set the first row to 1 so we could draw better graph starting at the original amount of dollars
    #to prices?
    ret_val = s_0* pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1 - 1 
    return ret_val

def helper_show_gbm(scenarios=3,port_mu=0.08,port_sigma=0.19,s0=100):
    """
    Draw the results of a stock price evolution under a Geometric Brownian Motion model
    """
    prices = gbm(n_scenarios=scenarios,mu=port_mu,sigma=port_sigma,s_0=s0)
    ax = prices.plot(legend=False,color="indianred",alpha=0.3,linewidth=2,figsize=(12,6)) #alpha is the transparency of the lines
    ax.axhline(y=s0,ls=":",color='black') # draw a line y = 100
    ax.set_xlim(left=0)
    ax.set_ylim(top=1000)
    #draw a dot the the origin
    ax.plot(0,s0,marker='o',color='darkred',alpha=0.2)

def show_gbm():
    """
    Run interactive simulation with GBM
    """
    gbm_controls = widgets.interactive(helper_show_gbm,
                                  scenarios=(1,20,1), # min, max, and stepside
                                  port_mu=(0,0.2,0.01),
                                  port_sigma = (0,0.3,0.01),
                                  s0 = (100,300,10)
                                  )
    display(gbm_controls)

def montecarlo(n_scenarios = 10,mu=0.07,
    sigma=0.15,original_value=100,m=3,floor=0.1,riskfree_rate=0.03,y_max=100):
   
    """
    Plot the results of a Monte Carlo simulation 
    """
    
    start = 100.0
    # constructing risky assets using simulator
    simple_rets = gbm(n_scenarios=n_scenarios,
                          mu=mu,
                          sigma=sigma,
                          s_0 = False,
                          steps_per_year=12,prices=False)
    risky_r = pd.DataFrame(simple_rets)
    
    #constructing ccpi
    brt= run_cppi(risky_r = pd.DataFrame(risky_r),
                      initial_value=start,
                      floor=floor,
                      riskfree_rate=riskfree_rate,
                      m=m,
                     )
    wealth = brt['Wealth']
    # use this logic to zoom in graphing 
    y_max = wealth.values.max() * y_max/100
    
    #plotting
    ax = wealth.plot(legend=False,
                     color="indianred",
                     alpha=0.3,
                     linewidth =2,
                     figsize=(12,6))
    #intial value
    ax.axhline(y=start,ls=":",color='black')
    # floor value 
    ax.axhline(y=start*floor,ls=":",color='red')
    ax.set_ylim(top=y_max)
    ax.set_xlim(left=0)
    ax.plot()

    
def show_montecarlo():
    """
    Run interactive simulation with montecarlo 
    """
    ccpi_controls = widgets.interactive(montecarlo,
                                  y_max=widgets.IntSlider(min=0,max=100,step=1,value=100,
                                                         description="Zoom Y Axis"),
                                  n_scenarios=widgets.IntSlider(min=1,max=100,step=5,value=50),
                                  mu=(0.0,0.2,0.01),
                                  sigma=(0,0.3,0.05),
                                  floor=(0,2,.1),
                                  m=(1,5,0.5),
                                  riskfree_rate=(0,0.05,.01),
                                  )
    display(ccpi_controls)

def discount(t,r):
    """
    Compute the price of a pure discount bond that pays a dollar at a time t, given interest rate r
    returns a dataframe
    """
    discounts = pd.DataFrame([(r+1)**-i for i in t])
    discounts.index = t
    return discounts #dataframe

def pv(flows,r):
    """
    computes the present value of a sequence of liabilities
    l is indexed by a the time, and the values are amounts of each liability
    returns a series
    """
    dates = flows.index
    discounts = discount(dates,r)
    return discounts.multiply(flows,axis='rows').sum() # series

def funding_ratio(assets,liabilities,r):
    """
    Computes the fund ratio of some assets given liabilities and interest rate
    """
    return pv(assets,r)/pv(liabilities,r)

def inst_to_ann(r):
    """
    convert short rate to annualize rate
    """
    return np.expm1(r)

def ann_to_inst(r):
    """convert annualized to a short rate"""
    return np.log1p(r)

def cir(n_years=3,n_scenarios=1,a=0.05,b=0.03,sigma=0.05,steps_per_year=12,r_0=None):
    """
    Implement the CIR model for interest rates
    """
    if r_0 is None:
        r_0 = b
    r_0 = ann_to_inst(r_0)
    dt = 1/steps_per_year
    
    num_steps = int(n_years * steps_per_year)+1
    shock = np.random.normal(0,scale=np.sqrt(dt),size=(num_steps,n_scenarios))
    # creating empty array with the same size of shock 
    rates = np.empty_like(shock)
    # first row is original price
    rates[0]=r_0
    
    """
    Implement the price of zero coupon bond
    """
    # For price gengeration
    h = math.sqrt(a**2 + 2*sigma**2)
    prices = np.empty_like(shock)
    
    def price(ttm,r):
        _A = ((2*h*np.exp((h+a)*ttm/2))/ (2*h+(h+a)*(np.exp(h*ttm)-1))) ** (2*a*b/sigma**2)
        _B = (2*(np.exp(h*ttm)-1)) / (2*h + (h+a)*(np.exp(h*ttm)-1))
        _P = _A*np.exp(-_B*r)
        return _P
    prices[0] = price(n_years,r_0)
    
    for step in range(1,num_steps):
        r_t = rates[step-1]
        d_r_t = a * (b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
        rates[step] = abs(r_t + d_r_t)
        prices[step]=price(n_years-step*dt,rates[step])
    
    rates=  pd.DataFrame(data=inst_to_ann(rates),index=range(num_steps))
    prices= pd.DataFrame(data=prices,index=range(num_steps))
    return rates,prices

def bond_cash_flows(maturity,principal=100,coupon_rate=0.03,coupons_per_year=12):
    """
    compute cashflows of bond
    """
    #number of coupons per year
    n_coupons = round(maturity *coupons_per_year)
    # coupon payment
    cash_flow = coupon_rate/coupons_per_year * principal
    cash_flows = np.repeat(cash_flow,coupons_per_year)
    #constructing index
    coupon_index = np.arange(1,n_coupons+1)
    #constructing Series of coupons
    cashflow_p = pd.Series(data=cash_flow,index=coupon_index)
    cashflow_p.iloc[-1] += principal
    return cashflow_p

def bond_price(maturity,principal=100,coupon_rate=0.03,coupons_per_year=12,discount_rate=0.03):
    """
    compute bond price
    """
    if isinstance(discount_rate,pd.DataFrame):
        pricing_dates = discount_rate.index
        prices = pd.DataFrame(index=pricing_dates,columns = discount_rate.columns)
        for t in pricing_dates:
            prices.iloc[t] = bond_price(maturity-t/coupons_per_year,principal,coupon_rate,coupons_per_year,
                                      discount_rate.iloc[t])
        return prices
    else:
        if maturity <=0 : return principal + principal*coupon_rate/coupons_per_year
        bond_cash_flow = bond_cash_flows(maturity=maturity,principal=principal,coupon_rate=coupon_rate,
                    coupons_per_year=coupons_per_year)
        bond_prices = pv(bond_cash_flow,discount_rate/coupons_per_year)
        return bond_prices

def macaulay_duration(flows,discount_rate):
    """
    computes the macaulay duration of a sequence of cashflows
    """
    discounted_flow = discount(flows.index,discount_rate).multiply(flows,axis='rows')
    m = (discounted_flow.index*discounted_flow[0]).sum()/discounted_flow.sum()
    return m[0]
    


def match_duration(liabilities,short_cf,long_cf,rate):
    """
    compute short term bond weight of a porfolio based on short term , 
    long term and target macaulay duration
    """
    target_w = macaulay_duration(liabilities,discount_rate=rate)
    short_w =  macaulay_duration(short_cf,discount_rate=rate)
    long__w =  macaulay_duration(long_cf,discount_rate=rate)
    return (target_w - long__w) / (short_w - long__w)

def bond_total_return(monthly_prices,principal,coupon_rate,coupons_per_year):
    """
    compute the total return of a bond based on bond prices and coupon payments
    """
    coupons = pd.DataFrame(data=0,index=monthly_prices.index,columns = monthly_prices.columns)
    t_max = monthly_prices.index.max()
    paydate = np.linspace(12/coupons_per_year,t_max,int(coupons_per_year * t_max/12),dtype=int)
    coupons.iloc[paydate] = coupon_rate/coupons_per_year * principal
    total_return = (monthly_prices + coupons)/monthly_prices.shift()-1
    return total_return.dropna()

def bt_mix(r1,r2,allocator,**kwargs):
    """
    Runs a back test of allocating between a two sets of returns
    r1 and r2 are T x N dataframe or returns where T is the timesteps and N is the # of scenarios
    allocator is a function that thakes 2 sets of returns and allocate specific parameter,s and produces
    an allocation to the first portfolio(the rest of the money is invested in GNP) as a Tx1 DataFrame
    Returns a Tx1 DataFrame of resulting N portfolio scenarios
    """
    if not r1.shape == r2.shape : 
        raise ValueError("r1 and r2 need to be the same shape")
    weights = allocator(r1,r2,**kwargs)
    if not weights.shape == r1.shape:
        raise ValueError("Allocator returned weights that dont match r1")
    r_mix = weights*r1 + (1-weights)*r2
    return r_mix 

def terminal_values(r):
    return (1+r).prod()

def fixedmin_allocator(r1,r2,w1,**kwargs):
    """
    Produces a time series over T steps of allocations between the PSP and GHP across N scenarios
    PSP and GHP are TxN DataFrames that represents the returns of PSP and GHP such that:
    each column is a scenario
    Returns a TxN DataFrame of PSP Weights
    """
    return pd.DataFrame(data=w1,index=r1.index,columns=r1.columns)
    

def terminal_stats(rets,floor = 0.8,cap=np.inf,name="Stats"):
    """
    Produce Summary Stats on the terminal values per invested dollar
    across a range of N scenarios
    rets is a Txn DataFrame of returns, where T is the time step
    Returns a 1 column DataFrame of Summary Stats indexed by the stat name
    """
    terminal_wealth = (1 + rets).prod() # (1+r1)(1+r2)... 
    breach = terminal_wealth < floor    # how many times breach happens
    reach = terminal_wealth >= cap
    p_breach = breach.mean() if breach.sum() > 0 else np.nan
    p_reach =  reach.mean() if reach.sum() > 0 else np.nan
    e_short = (floor - terminal_wealth[breach]).mean() if breach.sum()>0 else np.nan
    e_surplus = (cap - terminal_wealth[reach]).mean() if reach.sum() > 0 else np.nan
    sum_stats = pd.DataFrame.from_dict({
        "mean" : terminal_wealth.mean(),
        "std": terminal_wealth.std(),
        "p_breach": p_breach,
        "e_short":e_short,
        "p_preach":p_reach,
        "e_surplus": e_surplus
       
    },orient='index',columns=[name])
    return sum_stats

def glidepath_allocator(r1,r2,start_glide=1,end_glide=0):
    """
    Simulates a target_date- fund style gradual move from r1 to r2
    """
    n_points = r1.shape[0]
    n_col = r1.shape[1]
    path = pd.Series(data=np.linspace(start_glide,end_glide,num=n_points))
    paths = pd.concat([path]*n_col,axis=1)
    paths.index = r1.index
    path.columns = r1.columns
    return paths

def floor_allocator(psp_r,ghp_r,floor,zc_prices,m=3):
    """
    Allocate between PSP and GSP with the goal to provide exposure to the upside
    of the PSP without going violating the floor
    Uses a CPPI - style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP
    
    Returns a DataFrame with the same shape as the PSP/gsp representing the weights in the PSP
    """
    if zc_prices.shape != psp_r.shape:
        raise ValueError("PSP and ZC Prices must have the same shape")
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1,n_scenarios)
    floor_value = np.repeat(1,n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index,columns=psp_r.columns)
    
    for step in range(n_steps):
        floor_value = floor*zc_prices.iloc[step] ##PV of Floor assuming today'srate and flat YC
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0,1) # same as applying min and max. weight cant go above 1 and below 0 
        ghp_w = 1 - psp_w
        psp_alloc = account_value * psp_w
        ghp_alloc = account_value *ghp_w
        # recompute the new account value the end of this step
        account_value = psp_alloc*(1 + psp_r.iloc[step]) + ghp_alloc*(1 + ghp_r.iloc[step])
        w_history.iloc[step] = psp_w
    return w_history

def drawdown_allocator(psp_r,ghp_r,maxdd,m=3):
    """
    Allocate between PSP and GHP with goal to provide exposure to the upside
    of the PSP without going violating the floor
    Uses a CPPI style dynamic risk budgeting algorithm by investing a multiple of 
    the cushion in the PSP
    Returns a DataFrame with the same shape as the psp/ghp representing the weights in the PSP
    """
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1,n_scenarios)
    floor_value = np.repeat(1,n_scenarios)
    peak_value = np.repeat(1,n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index,columns=psp_r.columns)
    
    for step in range(n_steps):
        floor_value = (1 - maxdd)*peak_value ## Floor is based on prev peak
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0,1) # same as applying min and max. weight cant go above 1 and below 0 
        ghp_w = 1 - psp_w
        psp_alloc = account_value * psp_w
        ghp_alloc = account_value *ghp_w
        # recompute the new account value the end of this step
        account_value = psp_alloc*(1 + psp_r.iloc[step]) + ghp_alloc*(1 + ghp_r.iloc[step])
        peak_value = np.maximum(peak_value,account_value)
        w_history.iloc[step] = psp_w
    return w_history


def regress(dependent_variable,explantory_variables,alpha = True):
    """
    Runs a linear regression to decompose the dependent variable
    """
    if alpha : 
        explantory_variables = explantory_variables.copy()
        explantory_variables['Alpha'] = 1 
    
    lm = sm.OLS(dependent_variable,explantory_variables).fit()
    return lm.summary()

def tracking_error(r_a, r_b):
    """
    Returns the Tracking Error between the two return series
    """
    return np.sqrt(((r_a - r_b)**2).sum())

def style_analysis(dependent_variable, explanatory_variables):
    """
    Returns the optimal weights that minimizes the Tracking error between
    a portfolio of the explanatory variables and the dependent variable
    """
    n = explanatory_variables.shape[1]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    solution = minimize(portfolio_tracking_error, init_guess,
                       args=(dependent_variable, explanatory_variables,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    weights = pd.Series(solution.x, index=explanatory_variables.columns)
    return weights