import pandas as pd
import numpy as np

#same issuer, different trade date?
# x = df_merge_market_google.groupby('Issuer').size()
# x.loc[x>1]

#remove data and get a clean dataframe 
#google info only avaibale from 2004-01-01, has to exclude the previous 

df_raw = pd.read_csv('Data_clean/data_final_raw.csv')
df_raw['Trade_Date'] = pd.to_datetime(df_raw['Trade_Date'])
df_final = df_raw.loc[df_raw['Trade_Date'] >= '2004-01-01']
#-99 is missing in founding 
df_final['Founding'] = df_final['Founding'].replace(-99, np.nan)
df_final['Founding'].value_counts(dropna = False)
#remove IPOs with offer price less than 5
df_final = df_final.loc[df_final['Offer_Price'] >= 5] #28 gets removed 
df_final = df_final.rename(columns = {'1st Day.1_% Px Chng ' : '1st_Day_Return', '$ Change_Opening' : 'change_open', '$ Change.1_Close' : 'change_close'})
df_final['firm_age'] = df_final['Trade_Date'].dt.year - df_final['Founding']
# df_final.loc[df_final['firm_age'] <= 0]
#features
feature_y_col = ['1st_Day_Return','Star_Ratings','VC Dummy','Internet dummy',
                    'top_tier_uw','perc_price_above','ASVI','mean_SVI',
                    'week_-8','week_-7','week_-6','week_-5','week_-4','week_-3','week_-2','week_-1',
                    'firm_age','Star_Ratings']


df_final[feature_y_col].isnull().sum()
#drop any rows without vc or internet dummy, mostly error records in scoop
df = df_final.loc[ (df_final['VC Dummy'].notnull()) & (df_final['Internet dummy'].notnull())]
df[feature_y_col].isnull().sum()
#most company without age is from those are used to be excluded from the literature, drop them from our data set, 150 dropped
df = df.loc[df['firm_age'].notnull()]
df[feature_y_col].isnull().sum()
#drop only one without market hotness 
df = df.loc[df['perc_price_above'].notnull()]
#for anything related to google trend, if nan, label it 0 , only columns related with google trend has nan
df = df.fillna(0)

#lable nan star rating as 1

df['Star_Ratings'] = df['Star_Ratings'].replace('N/C', 1)

#replace VC dummy, as 1 
#just one 
df['VC Dummy'] = df['VC Dummy'].replace('.', 1) 
df['VC Dummy']= df['VC Dummy'].astype(int)
df['VC Dummy'] = df['VC Dummy'].replace(2, 1)
# df['VC Dummy'].value_counts()

#internet dummy 
df['Internet dummy'] = df['VC Dummy'].replace('.', 1) 
df['Internet dummy'].value_counts()
df = df.loc[df['Internet dummy'] != 9]

#label dummy myself

'''
df is the final data used for prediction 
'''
#symbol is not the unique identifier for IPO, but symbol and trade date combined is unique 

df = df.reset_index(drop= True).reset_index().rename(columns ={'index' : 'IPO_index'})



df.to_csv('Data_clean/Final_Train/IPO_train.csv' , index = False)

# df = pd.read_csv('Data_clean/Final_Train/IPO_train.csv')
# df = df.drop(columns = 'IPO_index')


# df['Star_Ratings'] = df['Star_Ratings'].replace('N/C', 1)


df = pd.read_csv('Data_clean/Final_Train/IPO_train.csv')
df['high_return'] = 0
df.loc[df['1st_Day_Return'] > df['1st_Day_Return'].mean() , 'high_return'] = 1


df['positive_return'] = 0
df.loc[df['1st_Day_Return'] >0 , 'positive_return'] = 1

df.to_csv('Data_clean/Final_Train/IPO_train.csv' , index = False)
