### road csv
import pandas as pd
import os

os.chdir('C:/Users/jstco/Downloads/data-cleansing-main/data-cleansing-main/Chapter01/data')
file_path = os.getcwd()
# print(file_path)

"""
 #   Column      Non-Null Count   Dtype
---  ------      --------------   -----
 0   locationid  100000 non-null  object
 1   year        100000 non-null  int64
 2   month       100000 non-null  int64
 3   temp        85554 non-null   float64
 4   latitude    100000 non-null  float64
 5   longitude   100000 non-null  float64
 6   stnelev     100000 non-null  float64
 7   station     100000 non-null  object
 8   countryid   100000 non-null  object
 9   country     99995 non-null   object
"""

# landtemps = pd.read_csv(file_path + '/landtempssample.csv')
# print(landtemps)

# landtemps = pd.read_csv(file_path + '/landtempssample.csv',
#                         names=['stationid', 'year', 'month', 'avgtemp', 'latitude',
#                                'longitude', 'elevation','station', 'countryid', 'country'],
#                         skiprows = 1,
#                         low_memory = False)

# landtemps.dropna(subset = ['avgtemp'], inplace = True)

from global_functions import DataAnalyzer
# analyzer = DataAnalyzer(landtemps)
# analyzer.summarize_basic()

### road excel
# percapitaGDP = pd.read_excel(file_path + '/GDPpercapita.xlsx',
#                              sheet_name = 'OECD.Stat export',
#                              skiprows = 4,
#                              skipfooter = 1,
#                              usecols = 'A, C:T')

# percapitaGDP.dropna(subset = ['2001'], inplace = True)
# analyzer = DataAnalyzer(percapitaGDP)
# analyzer.summarize_basic()

# percapitaGDP.rename(columns = {'Year':'metro'}, inplace = True)

# percapitaGDP.metro.str.startswith(' ').any()
# percapitaGDP.metro.str.endswith(' ').any()
# percapitaGDP.metro = percapitaGDP.metro.str.strip()
# print(percapitaGDP)

# for col in percapitaGDP.columns[1:]:
#     percapitaGDP[col] = pd.to_numeric(percapitaGDP[col], errors = 'coerce')
#     percapitaGDP.rename(columns = {col:'pcGDP' + col}, inplace = True)
#
# percapitaGDP.dropna(subset = percapitaGDP.columns[1:], how = 'all', inplace = True)

# print(percapitaGDP.head())
# print(percapitaGDP.describe())
# print(percapitaGDP.shape)
# print(percapitaGDP.metro.count())
# print(percapitaGDP.metro.nunique())

# for col in percapitaGDP.columns[1:]:
#     max = percapitaGDP[col].max()
#     min = percapitaGDP[col].min()
#     gap = max - min
#     print(f'max - min : {col} - {gap}')

# print(percapitaGDP.loc[2:])

### road database
import numpy as np
import pymssql
import sqlalchemy
import mysql.connector

# 방법 1. 데이터베이스 연결 객체 생성
# engine = sqlalchemy.create_engine('mssql+pymssql://pdccuser:pdccpass@pdcc.c9sqqzd5fulv.us-west-2.rds.amazonaws.com/pdcctest')
#
# # 쿼리 실행
# query = "SELECT studentid, school, sex, age, famsize,\
#   medu AS mothereducation, fedu AS fathereducation,\
#   traveltime, studytime, failures, famrel, freetime,\
#   goout, g1 AS gradeperiod1, g2 AS gradeperiod2,\
#   g3 AS gradeperiod3 From studentmath"
# result = pd.read_sql(query, engine)
# print(result)


## 방법 2
# server = "pdcc.c9sqqzd5fulv.us-west-2.rds.amazonaws.com"
# user = "pdccuser"
# password = "pdccpass"
# database = "pdcctest"
#
# engine = sqlalchemy.create_engine(f"mssql+pymssql://{user}:{password}@{server}/{database}")
#
# query = "SELECT studentid, school, sex, age, famsize,\
#   medu AS mothereducation, fedu AS fathereducation,\
#   traveltime, studytime, failures, famrel, freetime,\
#   goout, g1 AS gradeperiod1, g2 AS gradeperiod2,\
#   g3 AS gradeperiod3 From studentmath"
#
# studentmath = pd.read_sql(query, engine)
# # print(studentmath)
# print(studentmath.dtypes)

### load json
import json
import pprint
from collections import Counter

os.chdir(r'C:\Users\jstco\Downloads\data-cleansing-main\data-cleansing-main\Chapter02\data')
file_path = os.getcwd()
with open(file_path + '/allcandidatenewssample.json') as f:
  candidatenews = json.load(f)

# print(len(candidatenews))
# pprint.pprint(candidatenews[0:2])

