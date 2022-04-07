import pandas as pd
import os
df_price = pd.read_csv('Data_clean/IPO_return.csv')

'''
get the ipo info from SCOOP 
'''
path = 'Data/IPO_info'
file_path = os.listdir(path)
df_list = []
for f in file_path:
    df_f = pd.read_csv(f'{path}/{f}')
    df_list.append(df_f)

df_info = pd.concat(df_list)

cols = ['Industry','Employees', 'Founded','View Prospectus','Market Cap','Revenues','Net Income'
        ,'Symbol', 'Exchange','Shares (millions)','Price range','Est. $ Volume','Manager / Joint Managers'
        ,'CO-Managers','Expected To Trade' ,'Status']

df_info = df_info[cols]
df_info.drop_duplicates(subset = 'Symbol')
#match info and price using symbol
#not enough, also by year if multiple 
#use year of Trade date and Year of Expected to Trade 
df_info['Expected To Trade'] = pd.to_datetime(df_info['Expected To Trade'])
df_price['Trade_Date'] = pd.to_datetime(df_price['Trade_Date'])
df_info['year'] = df_info['Expected To Trade'].dt.year
df_price['year'] = df_price['Trade_Date'].dt.year
#first match by exact 
df = pd.merge(df_price,df_info,how = 'left', left_on = ['Symbol','year'] , right_on = ['Symbol','year'] )


df_new = pd.merge(df_price,df_info,how = 'left', left_on = 'Symbol' , right_on = 'Symbol' )


df.loc[df['Industry'].notnull()]
df_new.loc[df_new['Industry'].isnull()]
df_info.loc[df['Industry'].isnull()]


type(df_info['Expected To Trade'].iloc[1])


df.loc[df['Expected To Trade'].notnull()]






df_price.loc[df_price['Symbol'] == 'ADCI']
df_info.loc[df_info['Symbol'] == 'ADCI']
df_new.loc[df_new['Symbol'] == 'NOVA']

'''
it doesn't match well
'''
df_age = pd.read_excel('Data/IPO-age.xlsx', usecols= "A:K")
df_merge = pd.merge(df_price,df_age,how = 'left', left_on = 'Symbol', right_on = 'Ticker')

df_merge.loc[df_merge['Internet dummy'].isnull()]


df_infomerge = pd.merge(df_info,df_age,how = 'left', left_on = 'Symbol', right_on = 'Ticker')
df_infomerge.loc[df_infomerge['Internet dummy'].isnull()]



#read summary first
df_sum = xl.parse(0, skiprows= 3, skipfooter= nrows- 25).dropna(axis=1, how='all')
df_sum = df_sum.rename(columns ={'Unnamed: 4' : 'Missed'})






df_price.loc[df_price['Symbol'] == 'AACQU']
df_price.to_csv('price.csv',index = False)

#! Some symbols are duplciated in df_info
merg

df_info.loc[df_info['Symbol'] == 'AACQU']



x = df_info.groupby('Symbol').size()
x.loc[x>1]
x.loc[x['']]
df_price['Symbol']


x = df_price.groupby('Symbol').size()

df_info.loc[df_info['Symbol'] == 'AHI']['View Prospectus']
df_price.loc[df_price['Symbol'] == 'AHI']

