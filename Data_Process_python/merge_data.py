import pandas as pd
import os
df_price = pd.read_csv('Data_clean/IPO_return.csv')
df_price['Trade_Date'] = pd.to_datetime(df_price['Trade_Date'])



df_age = pd.read_excel('Data/IPO-age.xlsx', usecols= "A:K",converters= {'Offer Date' : str} )
df_age['Offer Date'] = pd.to_datetime(df_age['Offer Date'], format = '%Y%m%d')

df_age['year'] = df_age['Offer Date'].dt.year
df_price['year'] = df_price['Trade_Date'].dt.year
'''
there are some error, duplicate IPO with different price, delete them 
'''
x = df_age.groupby(['Ticker','year']).size()
x.loc[x>1]
ticker_error = ['BSTZ','PPACU','VIOT','XONE']
df_price = df_price.loc[~df_price['Symbol'].isin(ticker_error)]

#merge in dummies and year from Ritter
df_merge = pd.merge(df_price,df_age,how = 'left', left_on = ['Symbol','year'], right_on = ['Ticker','year'])

'''
get top-tier dummy
'''
#use the max rank of all underwriters of an IPO 

def get_max_rank(row , uw_match_df):
    uw_list = row['Lead/Joint-Lead_ Managers'].split( '/')
    year = row['year']
    x  = [i for i in uw_match_df.columns if i.startswith('Rank') ]
    y = [i.replace('Rank', '') for i in x ]
    z = [ [i[:2], i[2:]] for i in y ]
    c = [ f'19{i}' if int(i) > 50 else f'20{i}' for sub in z for i in sub ]
    d = [c[n:n+2] for n in range(0, len(c), 2)]
    for index, item in enumerate(d):
        if (int(item[0]) <= year) and (int(item[1]) >= year):
            rank_col = x[index]
            break
    max_rank = 0
    for uw in uw_list:
        try:
            uw_rank =uw_match_df.loc[uw_match_df['uw_price'] == uw][rank_col].iloc[0]
        except:
            uw_rank = np.nan
            print(f'{uw},{year}')
        if uw_rank > max_rank:
            max_rank = uw_rank
    return max_rank



uw_match = pd.read_csv('Data_clean/uw_match.csv')
df_price.columns
df_merge.columns

# df_price['uw_max_rank'] = df_price.apply(get_max_rank, uw_match_df =uw_match , axis = 1 )
df_merge['uw_max_rank'] = df_merge.apply(get_max_rank, uw_match_df =uw_match , axis = 1 )
#get top-tier dummy 

df_merge['top_tier_uw'] = 0
df_merge.loc[df_merge['uw_max_rank'] >= 8 , 'top_tier_uw'] = 1


'''
get market hotness
'''

market_hot  = pd.read_excel('Data/IPOALL.xlsx', usecols= "A:F" , header=None, skipfooter= 5)
market_hot.columns = ['month','year','avg_net_ipo','num_ipo','net_num_ipo','perc_price_above']
market_hot = market_hot.loc[market_hot['year'] < 22 ]
market_hot['year'] = market_hot.apply(lambda x : int(f"20{x['year']}") if len(str(x['year'])) ==2 else int(f"200{x['year']}"), axis = 1 )
#convert number to percentage 
market_hot['perc_price_above'] = pd.to_numeric(market_hot['perc_price_above'], errors='coerce') / 100
#merge it with df_merge, on year and month 

df_merge['month'] = df_merge['Trade_Date'].dt.month
df_merge_market = pd.merge(df_merge,market_hot,how = 'left', on= ['year', 'month'])
df_merge_market.loc[df_merge_market['perc_price_above'].isnull()]

# df_merge_market.to_csv('Data_clean/data_train.csv', index = False)


df_merge_market['Issuer'] =  df_merge_market['Issuer'].str.rstrip()
'''
merge google trend data
'''
google_svi = pd.read_csv('Data_clean/google_svi_index.csv')
google_svi['Trade_Date'] = pd.to_datetime(google_svi['Trade_Date'])

df_merge_market_google = pd.merge(df_merge_market,google_svi, how = 'left', on = ['Issuer','Trade_Date']  )

df_merge_market_google.to_csv('Data_clean/data_final_raw.csv', index = False)

