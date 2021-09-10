import pandas as pd
import edhec_risk_kit as erk
import numpy as np
import math
import ipywidgets as widgets
import numpy_financial as npf
import re
import colorama
from colorama import Fore, Style
import string
from termcolor import colored


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
                                  maturity=widgets.FloatText(min=1,max=100,step=1,value=4,description='Maturity'),
                                  principal=widgets.FloatText(min=100,max=10000,step=100,value=1000,description='Principal'),
                                  couponrate=widgets.FloatText(min=0.0,max=1,step=0.01,value=0.05, description='Coupon Rate'),
                                  coupon_per_year=widgets.Dropdown(options=[('Zero coupon',0),('Annually',1),('Semi Annually',2),
                                                                            ('Quarterly',4),('Monthly',12)],value=2,description='# of coupons'),
                                  ytm=widgets.FloatText(min=0.0,max=1,step=0.01,value=0.05, description='YTM'))
                                                                   
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
            str1 = Fore.BLUE + "Zero coupon bond price is: " +  "${:,.2f}".format(bond_prices)
    # coupons bond
    else :  
        bond_cash_flow = erk.bond_cash_flows(maturity=maturity,principal=principal,coupon_rate=coupon_rate,
                        coupons_per_year=coupons_per_year)
        bond_prices = erk.pv(bond_cash_flow,discount_rate/coupons_per_year).round(2)
    
        str1 = Fore.BLUE + "\n" + "The price of the " + str(maturity) + " year(s) bond is : " +  "${:,.2f}".format(bond_prices[0])
    
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
                                  maturity=widgets.FloatText(min=1,max=40,step=1,value=4,description='Maturity'),
                                  principal=widgets.FloatText(min=100,max=10000,step=100,value=1000,description='Principal'),
                                  coupon_rate=widgets.FloatText(min=0.0,max=1,step=0.01,value=0.05, description='Coupon Rate'),
                                  coupons_per_year=widgets.Dropdown(options=[('Zero coupon',0),('Annually',1),('Semi Annually',2),
                                                                            ('Quarterly',4),('Monthly',12)],value = 2, description = "# of coupons"),
                                                                    
                                  discount_rate=widgets.FloatText(style={'description_width': 'initial'},min=0.0,max=0.5,step=0.01,value=0.05, description='Interest Rate'))
                                                                 
                                                                  
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
                                      maturity=widgets.FloatText(min=1,max=40,step=1,value=4,description='Maturity'),
                                      par_value=widgets.FloatText(min=100,max=10000,step=100,value=1000,description='Par Value'),
                                      coupon_rate=widgets.FloatText(min=0.01,max=0.2,step=0.1,value=0.03,description='Coupon Rate'),
                                      coupons_per_year=widgets.Dropdown(options=[('Zero coupon',0),('Annually',1),('Semi Annually',2),
                                                                            ('Quarterly',4),('Monthly',12)],value = 2,description = "# of coupons"), 
                                      bond_price=widgets.FloatText(min=0,max=2000,step=1,value=100,description='Bond Price'))
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
    
    gcontrols =   widgets.interactive(convexity,
                                      maturity=widgets.FloatText(min=1,max=40,step=1,value=4,description='Maturity'),
                                      principal=widgets.FloatText(min=100,max=10000,step=100,value=1000,description='Principal'),
                                      couponrate=widgets.FloatText(min=0.01,max=0.2,step=0.1,value=0.03,description='Coupon Rate'),
                                      coupons_per_year=widgets.Dropdown(options=[('Zero coupon',0),('Annually',1),('Semi Annually',2),
                                                                            ('Quarterly',4),('Monthly',12)],value = 2,description = "# of coupons"),
                                      discount_rate=widgets.FloatText(style={'description_width': 'initial'},min=0.0,max=0.5,step=0.01,value=0.05, 
                                                                      description='Interest Rate'))
                                     
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
                                     x=widgets.Dropdown(options=[('Macaulay Duration',1),
                                                                 ('Price',2),
                                                                 ('YTM',3),
                                                                 ('Convexity',4)],
                                                                 description='Solve for'),
                                     )
    display(control)
    