import pandas as pd
from fuzzywuzzy import fuzz

def match_uw(uw_price, uw_rank ):
    match_dict = {}
    match_dict['uw_price'] = []
    match_dict['uw_rank'] = []
    match_dict['socre'] = []
    i = 0
    for uw_p in uw_price:
        max_score = 0 
        matched = None
        for uw in uw_rank:
            score = fuzz.token_set_ratio(uw_p,uw)
            if score > max_score:
                matched = uw
                max_score = score 
        match_dict['uw_price'].append(uw_p)
        match_dict['socre'].append(max_score)
        match_dict['uw_rank'].append(matched)
        i += 1
        print(f"{i} is done")
    return match_dict


'''
match underwritter
'''
uw_rank = pd.read_excel('Data/Underwriter-Rank.xls', skipfooter= 2 ).dropna(axis=1, how='all')
#columns means year 
df_price = pd.read_csv('Data_clean/IPO_return.csv')


#get the list of unique underwritter, and then do fuzzy match 
uw_list = list(df_price['Lead/Joint-Lead_ Managers'].str.split( '/').values)
uw_list = [item for sublist in uw_list for item in sublist]
uw_list = list(set(uw_list))
#for each, find the similarities 

uw_rank_list = list(set(uw_rank['Underwriter Name'].values))

matched = match_uw(uw_price = uw_list , uw_rank = uw_rank_list) 
matched_df = pd.DataFrame.from_dict(matched)
matched_df = matched_df.loc[matched_df['uw_price'] != '']
merge = pd.merge(matched_df,uw_rank, how = 'left', left_on = 'uw_rank' , right_on = 'Underwriter Name' )
merge.to_csv('Data_clean/uw_match.csv',index = False)


