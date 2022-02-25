#!/usr/bin/python
import pandas as pd


name_stocks = ['AAPL','ABB','ABBV','TOT','WMT','DUK','CHL','HSBC']

for name in name_stocks: 
	df = pd.read_csv(name+'_error.csv')
	print(df.mean(axis=0))
	
