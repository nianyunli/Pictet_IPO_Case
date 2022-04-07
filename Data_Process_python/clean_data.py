import pandas as pd

xl = pd.ExcelFile('Data/SCOOP-Rating-Performance.xls')
nrows = xl.book.sheet_by_index(0).nrows

#read summary first
df_sum = xl.parse(0, skiprows= 3, skipfooter= nrows- 25).dropna(axis=1, how='all')
df_sum = df_sum.rename(columns ={'Unnamed: 4' : 'Missed'})


##tables are seperated by rows of nan value 
#read date as string
df_return = xl.parse(0, skiprows= 34 , converters= {xl.parse(0 ,skiprows= 34).columns[0]: str}).dropna(axis=1, how='all')
df_return.columns = pd.MultiIndex.from_arrays([df_return.columns, df_return.iloc[0]])
df_return.columns = ['_'.join(col) if not col[0].startswith('Unnamed') else col[1] for col in df_return.columns.values ]

#drop duplicate rows 
df_price = df_return.drop(df_return.loc[df_return['Trade_Date'] == 'Date'].index , axis = 0)
df_price = df_price.drop(df_price.loc[df_price['Trade_Date'] == 'Trade'].index , axis = 0)
df_price = df_price.drop(df_price.loc[df_price['Trade_Date'].isnull()].index , axis = 0)

#convert to datetime
#the date is incorret for company Alon USA Partners, LP, not able to guess the exact day
df_price['Trade_Date'] = pd.to_datetime(df_price['Trade_Date'] , errors = 'coerce')

df_price.to_csv('Data_clean/IPO_return.csv', index = False)
# df_price['year'] = df_price['Trade_Date'].dt.year

'''
end of script
'''
# df_price.loc[df_price['Trade_Date'].isnull()]



# #seperate lead and joint lead manager
# x = df_price['Lead/Joint-Lead_ Managers'].str.split( '/')
# y = x.apply(lambda x : len(x))
# x.apply(lambda x : len(x)).max()

# y.loc[y ==15]

# df_price.loc[]

# y = pd.to_datetime(df_price['Trade_Date'].str.split(' ').str[0] , errors = 'coerce')
# y.loc[y.isnull()]
# x.iloc[1] == y.iloc[1]

# print(x.iloc[1])
# print(type(y.iloc[1]))

# y.loc[y == '2020-01-17']

# df_price.loc[1811]


# df_price.loc[df_price['Trade_Date'] == '11/120']

# pd.to_datetime(df_price['Trade_Date'].str.split(' ').str[0], errors = 'coerce')
# df_return.iloc[1811]
# df_price.iloc[1811]

# df_price.loc[]


# x = pd.to_datetime(df_price['Trade_Date'], errors = 'coerce')

# x.loc[x.isnull()]

# df_price.iloc[1811]['Trade_Date']



# df_price.loc[df_price['Trade_Date'].str.contains('120-11-01')]
# df_price.loc[df_price['Trade_Date'].isnull()]

# df_price.to_csv('test.csv',index = False)

# type(df_price['Trade_Date'].iloc[0])


# df_price.loc[df_price['Trade_Date'] == 'Trade']

# # df_return.loc[df_return['Trade_Date'] == 'Date']

# # df_price = df_return.drop_duplicates()
# df_price = df_price.drop(df_price.loc[df_price['Trade'] == 'Trade'].index , axis = 0)
# df_price = df_price.drop(df_price.loc[df_price['Trade'] == 'Trade'].index , axis = 0)

# type(df_return['Trade'].iloc[3])



# df_price.loc[df_price['Trade'] == 'Trade']
# x = df_return.drop_duplicates()
# df_return.columns

# df_return.drop_duplicates().loc[df_return['Trade'] == 'Trade']

# df_return.loc[]

# df_return.loc[df_return.isnull().all(1)]


# x.loc[x.isnull().all(1)]
