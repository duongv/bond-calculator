import pandas as pd
import numpy as np

def read_file(file_csv):
    """
    Read and clean csv file.Return a dataframe
    """
    df = pd.read_csv(file_csv,index_col=0,header=0,na_values = '-')
    df.loc['Equity'] = df.loc['Total Assets'] - df.loc['Total Liabilities']
    return df

def average_consecutive(row):
    """
    Compute the average of 2 consecutive values in a row of a dataframe
    Return a list  
    """
    k = []
    # zip values to find average of 2 consecutive values. 
    for a,b in zip(row,row[1::1]): 
        avg = (a+b)/2
        k.append(avg)
    k.append(np.nan) # append nan at the end cause the last year shouldnt have average
    return k

def empty_row(x,df):
    """
    Create an empty row for dataframe - same column names etc
    return a empty dataframe 
    x is name index, df is the original dataframe
    """    
    empty_list= []
    empty_list.extend([np.nan for i in range(df.shape[1])])  # create a list of n/a. the size of the list is the same as df.colums.size
    empty_df = pd.DataFrame([empty_list],index=[x],columns=df.columns)
    return empty_df
    
def ratio(file_csv):
    
    df = read_file(file_csv) # get the df 
    """
    Dupont ratio
    """
    # return on sales 
    df.loc['Return on Sales'] = df.loc['Net Income'] / df.loc['Revenue']
    #assets turn over
    df.loc['Assets Turnover'] = df.loc['Revenue'] / df.loc['Total Assets']
    #financial leverage
    financial_leverage = []
    for i in range(len(average_consecutive(df.loc['Total Assets']))):
        financial_leverage.append(average_consecutive(df.loc['Total Assets'])[i] / average_consecutive(df.loc['Equity'])[i])
    f= 'Financial Leverage' + ' (Assets/Equity' + ')' 
    df.loc[f] = financial_leverage
    dupont = df.loc[['Return on Equity (ROE)','Return on Assets (ROA)',f,'Return on Sales','Assets Turnover']]
    dupont = dupont[dupont.columns[::-1]]
    dupont['Average'] = dupont.mean(axis=1)
    """
    Profit Margin
    """
    df.loc['SG&A to Sales'] = df.loc['Selling, General & Admin'] / df.loc ['Revenue']
    df.loc['Effective Income Tax'] = df.loc['Income Tax'] / df.loc['Pretax Income'] #effective tax rate
    profit_margin = df.loc[['Gross Margin','SG&A to Sales','Operating Margin','Effective Income Tax']]
    profit_margin = profit_margin[profit_margin.columns[::-1]] # reverse columns 
    profit_margin['Average'] = profit_margin.mean(axis=1) # columns mean
       
    # Turnover ratios 
    df.loc['Days Receivables'] = 365 / (df.loc['Revenue'] / average_consecutive(df.loc['Receivables']))  # 365 / acct receivable turnover
    df.loc['Days Inventory'] = 365 / (df.loc['Cost of Revenue'] / average_consecutive(df.loc['Inventory'])) # 365 / inv turnover
    df.loc['Fixed Asset Turnover'] = (df.loc['Revenue'] / average_consecutive(df.loc['Property, Plant & Equipment'])) 
    
    purchases = df.loc[['Inventory','Cost of Revenue']]
    k = []
    for i in range(0,purchases.shape[1]-1):
        k.append(purchases.iloc[0,i] + purchases.iloc[1,i] - purchases.iloc[0,i+1]) # purchases = ending inv + cogs - beg inv
    k.append(np.nan)
    df.loc['Purchases'] = k 
    df.loc['Days Payable'] = 365 / (df.loc['Purchases']/ average_consecutive(df.loc['Accounts Payable']))
    turnover = df.loc[['Days Receivables','Days Inventory','Days Payable','Fixed Asset Turnover']]
    turnover = turnover[turnover.columns[::-1]] # reverse columns 
    turnover['Average'] = turnover.mean(axis=1) # compute columns mean
    
    #liquidity ratios 
    df.loc['Quick Ratio'] = df.loc['Cash & Cash Equivalents'] / df.loc['Total Current Liabilities'] 
    df.loc['CFO / Current Lib'] = df.loc['Operating Cash Flow'] / average_consecutive(df.loc['Total Current Liabilities'])
    df.loc['Long term debt / Equity'] = df.loc['Total Long-Term Liabilities'] / df.loc['Equity']
    df.loc['Long term debt / Tangibles'] = df.loc['Total Long-Term Liabilities'] / (df.loc['Total Assets'] - df.loc['Goodwill and Intangibles'])
    liquidity = df.loc[['Current Ratio','Quick Ratio','CFO / Current Lib','Long term debt / Equity','Long term debt / Tangibles']]
    liquidity = liquidity[liquidity.columns[::-1]]
    liquidity['Average'] = liquidity.mean(axis=1)
    
    # concatenate dataframes along with empty dataframes 
    final = pd.concat([empty_row('DUPONT',dupont),dupont,empty_row('PROFITABILITY',profit_margin),profit_margin,empty_row('TURN OVER',turnover),
                       turnover,empty_row('LIQUIDITY',liquidity),liquidity])
    return final

def commonsize_income_statement(file_csv):
    """
    Compute common size income statement as a df
    """ 
    df = read_file(file_csv)
    elements = ['Revenue','Cost of Revenue','Gross Profit','Selling, General & Admin','Research & Development',
               'Operating Expenses','Operating Income','Other Expense / Income','Pretax Income','Income Tax','Net Income']
    for i in elements:  # check if element is in the income statement. if not, return n/a rows .
        if i not in df.index:
            df.loc[i] = np.nan
    income = df.loc[elements]
    commonsize_income = income / df.loc['Revenue'] # devide values by revenue
    commonsize_income = commonsize_income[commonsize_income.columns[::-1]] # reverse the columns 
    commonsize_income['Average'] = commonsize_income.mean(axis=1) # calculate average
    return round(commonsize_income,2)

def commonsize_balance_sheet(file_csv):
    """
    Compute the common size balance sheet.
    Input is csv file. 
    Return a dataframe
    """
    df = read_file(file_csv) # read cs
    assets= ['Cash & Cash Equivalents','Receivables','Inventory','Other Current Assets','Total Current Assets','Property, Plant & Equipment',
               'Goodwill and Intangibles','Total Long-Term Assets','Total Assets']
    liabilities_se = ['Accounts Payable','Deferred Revenue','Current Debt',
               'Other Current Liabilities','Total Current Liabilities','Long-Term Debt','Other Long-Term Liabilities','Total Liabilities','Equity',
               'Total Liabilities and Equity']
    for i in assets + liabilities_se:  # if element not in index, the row is added as n/a values
        if i not in df.index:
            df.loc[i] = np.nan
    
    bs = pd.concat([empty_row('ASSETS',df),df.loc[assets],empty_row('LIABILITIES AND EQUITY',df),df.loc[liabilities_se]]) # assets + empty rows + liabilites/se
    bs = bs / df.loc['Total Assets']  # divide by total assets to get common size values 
    bs = bs[bs.columns[::-1]] # reverse columns 
    bs['Average'] = bs.mean(axis=1) # calculate average value
    bs = bs.round(2)
    bs.fillna(value=' ',inplace=True)
    return bs