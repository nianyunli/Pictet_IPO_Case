import pandas as pd
import os
from datetime import timedelta
from datetime import date
import numpy as np
import math
path = 'Data/Google_trend'
file_path = os.listdir(path)
df_list = []
for f in file_path:
    df_f = pd.read_csv(f'{path}/{f}')
    if len(df_f) > 0:
        df_list.append(df_f)

df_google = pd.concat(df_list)


df_trend = pd.read_csv('/Users/nianyun/Documents/Pictet_IPO_Case/Data_clean/IPO_return.csv')
df_trend = df_trend.loc[ (df_trend['Issuer'].notnull()) & (df_trend['Trade_Date'].notnull())]
df_trend['Issuer'] = df_trend['Issuer'].str.rstrip()
# len(df_trend['Issuer'].unique())
# x = df_trend.groupby(['Issuer','Trade_Date']).size()
# x.loc[x>1]
# df_trend.loc[df_trend['Issuer'] == 'Adial Pharmaceuticals']

company_date = dict(zip(df_trend.Issuer, df_trend.Trade_Date))
# company_date['BlackRock Science and Technology Trust II']


#get 8 weeks before and 8 weeks after for an IPO, and IPO week is week 0 
def get_date(row):
    issuer = row['Issuer']
    date = company_date[issuer]
    return date

df_google['Trade_Date'] = df_google.apply(get_date, axis = 1)
df_google['Trade_Date'] = pd.to_datetime(df_google['Trade_Date'])
df_google['date'] = pd.to_datetime(df_google['date'])


# (datetime.strptime('2015-01-04', '%Y-%m-%d') + timedelta(days=1)).isocalendar()
# (datetime.strptime('2015-01-11', '%Y-%m-%d') + timedelta(days=1)).isocalendar()

# datetime.strptime('2015-01-09', '%Y-%m-%d').strftime("%U")


def get_week(row, date_col):
    date = row[date_col] + timedelta(days=1)
    year, week_num, day_of_week = date.isocalendar()
    return week_num
def get_day(row, date_col):
    date = row[date_col] + timedelta(days=1)
    year, week_num, day_of_week = date.isocalendar()
    return day_of_week

df_google['Trade_week'] = df_google.apply(get_week,date_col = 'Trade_Date', axis = 1) 
df_google['Trade_day_of_week'] = df_google.apply(get_day,date_col ='Trade_Date', axis = 1) 
df_google['trend_week'] = df_google.apply(get_week,date_col = 'date', axis = 1) 
#relative week
def cal_relative_week(row):
    def weeks_for_year(year):
        last_week = date(year, 12, 28)
        return last_week.isocalendar()[1]
    year_trend = row['date'].year
    year_trade = row['Trade_Date'].year
    week_trend = row['trend_week']
    week_trade = row['Trade_week']
    if (row['Trade_Date'] > row['date']) and (week_trend >week_trade ):
        no_week = weeks_for_year(year_trend)
        relative_week  = -(no_week + week_trade - week_trend)
    elif (row['Trade_Date'] < row['date']) and (week_trade >week_trend ):
        no_week = weeks_for_year(year_trade)
        relative_week  = no_week - week_trade + week_trend
    else:
        relative_week  = week_trend - week_trade
    return relative_week 

df_google['relative_week'] = df_google.apply(cal_relative_week, axis = 1)



#abnormal SVI (ASVI)

def cal_ASVI(df ,base = -1):
    '''
    definition in the paper 
    groupby issuer, and calculate 
    '''
    svi_base = df.loc[df['relative_week'] == base]['SVI'].iloc[0]
    previous_8 = np.arange(base-8,base)
    previous_8 = np.asarray([df.loc[df['relative_week'] == i]['SVI'].iloc[0] for i in previous_8 ])
    median = np.median(previous_8)
    asvi = math.log(svi_base + 0.00001) - math.log(median + 0.00001)
    return asvi

def cal_mean(df):
    '''
    definition in the paper 
    groupby issuer, and calculate 
    '''
    mean = np.arange(-8,0)
    mean = np.asarray([df.loc[df['relative_week'] == i]['SVI'].iloc[0] for i in mean ])
    mean = np.mean(mean)
    return mean


asvi_df = df_google.groupby('Issuer').apply(cal_ASVI)
asvi_df = asvi_df.reset_index().rename(columns= {0: 'ASVI'})

mean_df = df_google.groupby('Issuer').apply(cal_mean)
mean_df = mean_df.reset_index().rename(columns= {0: 'mean_SVI'})

pivot_df = df_google.pivot_table('SVI', ['Issuer'], 'relative_week')
pivot_df.columns = [f'week_{col}'for col in pivot_df.columns]
cols = [col for col in pivot_df.columns if int(col.split('_')[-1]) in np.arange(-9,10)]
pivot_df = pivot_df[cols]
pivot_df = pivot_df.reset_index()

svi_index_df = pd.merge(asvi_df,mean_df, how ='outer', left_on = 'Issuer', right_on = 'Issuer' )
svi_index_df = pd.merge(svi_index_df,pivot_df, how ='outer', left_on = 'Issuer', right_on = 'Issuer' )
svi_index_df['Trade_Date'] = svi_index_df.apply(get_date, axis = 1)
# svi_index_df['Trade_Date'] = pd.to_datetime(svi_index_df['Trade_Date'])

# asvi_df.to_csv('Data_clean/asvi_df.csv', index = False)
svi_index_df.to_csv('Data_clean/google_svi_index.csv', index = False)
df_google.to_csv('Data_clean/google_trend_df.csv', index = False)



# x = df_google.groupby(['Issuer','Trade_Date']).size()
# x.loc[(x!=52) & (x!=53)]

# len(asvi_df.loc[asvi_df < 0])

# df_trend.loc[df_trend['Issuer'] == 'salesforce.com']

# #simply using 8 weeks data from 

# x = df_google.loc[df_google['Issuer'] == 'Advanced Life Sciences']