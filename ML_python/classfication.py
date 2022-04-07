from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error 
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import argparse
import pickle

parser = argparse.ArgumentParser(description='train_data')
parser.add_argument('--outcome', type=str,help='outcome column')
parser.add_argument('--model', type=str,help='which model')
parser.add_argument('--seed', type=int,help='which seed')
args = parser.parse_args()

class Preprocess:
    def __init__(self):
        self.original_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_all = None
        self.scaler = None
    def pre_process(self,df,outcome,features, test_size, seed ):
        '''
        encode categorical variables, split training data
        df is the whole df before prediction
        outcome is the outcome variable
        features are the fetires used for prediction
        '''
        self.original_df = df.copy()
        cols = [outcome,*features]
        self.train_df = df[cols]
        y = self.train_df.loc[:,outcome] 
        X = self.train_df.loc[:,self.train_df.columns != outcome]
        self.X_all = X.copy()
        #split data, 30% test data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state= seed) 
        #standardize it 
        dummy = ['VC Dummy','Internet dummy' , 'top_tier_uw']
        stand_col = [col for col in features if col not in  dummy]
        self.scaler = preprocessing.StandardScaler().fit(self.X_train[stand_col])
        transformed  = pd.DataFrame(self.scaler.transform(self.X_all[stand_col]), columns = stand_col)
        self.X_all[stand_col] = transformed  
        self.X_train = self.X_all.loc[self.y_train.index]
        self.X_test = self.X_all.loc[self.y_test.index]
        #label test data
        self.original_df['is_test'] = 0 
        self.original_df.loc[self.y_test.index,'is_test'] = 1 
        self.original_df['is_train'] = 0 
        self.original_df.loc[self.y_train.index,'is_train'] = 1 

class Prediction:
    def __init__(self):
        self.original_df = None
        self.accuracy = None 
        self.roc_auc_score = None
        self.model = None
        self.best_score_cv = None
        self.F_1 = None
        self.FPR = None
        self.FNR = None
    # def ridge_train(self,X_train,y_train):
    #     ridge_reg = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10], cv= 10 , scoring = 'neg_mean_squared_error')
    #     ridge_reg.fit(X_train, y_train)
    #     self.best_score_cv = ridge_reg.best_score_
    #     return ridge_reg
    # def ridge_pred(self,original_df,X_test,y_test,X_all,ridge_reg):
    #     self.original_df = original_df.copy()
    #     self.model = ridge_reg
    #     y_pred = ridge_reg.predict(X_test)
    #     y_pred_all = ridge_reg.predict(X_all)
    #     self.original_df['prediction'] = pd.Series(y_pred_all,index = X_all.index)
    #     self.test_MSE = mean_squared_error(y_test, y_pred)
    #     self.test_RMSE = np.sqrt(self.test_MSE)
    def xgb_tune(self,X_train,y_train,params):
        reg_xgb = xgb.XGBClassifier(objective = 'binary:logistic',use_label_encoder= False, eval_metric= 'auc')
        random_search = RandomizedSearchCV(reg_xgb ,n_iter = 40, param_distributions =params, cv= 10,verbose=0,scoring = 'roc_auc' )
        random_search.fit(X_train,y_train)
        reg_xgb = random_search.best_estimator_ 
        #* add best score from cv
        self.best_score_cv = random_search.best_score_
        return reg_xgb
    def xgb_train(self,reg_xgb,X_train,y_train): 
        reg_xgb.fit(X_train,y_train)
        return reg_xgb
    def xgb_pred(self,original_df,X_test,y_test,X_all,reg_xgb):
        self.original_df = original_df.copy()
        self.model = reg_xgb
        y_pred = reg_xgb.predict(X_test)
        y_pred_prob = reg_xgb.predict_proba(X_all)[:,1]
        y_pred_all = reg_xgb.predict(X_all)
        self.original_df['prediction'] = pd.Series(y_pred_all,index = X_all.index)
        self.original_df['predic_proba'] = pd.Series(y_pred_prob,index = X_all.index)
        ind = y_test.index
        y_pred_prob = self.original_df.loc[self.original_df['is_test'] ==1,'predic_proba']
        self.accuracy= accuracy_score(y_test, y_pred)
        self.roc_auc_score = roc_auc_score(y_test, y_pred_prob.reindex(ind))
        self.F_1 = f1_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test,y_pred ).ravel()
        self.FPR = fp / (fp+tn)
        self.FNR = fn / (fn+tp)




def pipeline_clf(outcome, features, model ,seed):
    df = pd.read_csv('/Users/nianyun/Documents/Pictet_IPO_Case/Data_clean/Final_Train/IPO_train.csv')
    # df['high_return'] = 0
    # df.loc[df['1st_Day_Return'] > df['1st_Day_Return'].mean() , 'high_return'] = 1
    process = Preprocess()
    process.pre_process(df = df,outcome = outcome,features = features,test_size =0.2, seed = seed)
    X_train = process.X_train
    y_train = process.y_train
    X_test  = process.X_test
    y_test  = process.y_test
    X_all = process.X_all
    df_labeled = process.original_df
    # if model == 'linear':
    #     linear = Prediction()
    #     ridge_reg = linear.ridge_train(X_train =X_train ,y_train = y_train)
    #     linear.ridge_pred(original_df =df_labeled ,X_test =X_test ,y_test =y_test ,X_all = X_all,ridge_reg =ridge_reg )
    #     return(process, linear)
    if model == 'xgboost':
        params = {"learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
                    "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
                    "min_child_weight" : [ 1, 3, 5, 7 ,10],
                    "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4,1,1.5],
                    "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7, 0.9,1 ],
                    "n_estimators"     : [10,50, 150, 200, 250, 300,350,400],
                    # "reg_alpha"        : [1e-5, 1e-4,1e-3,1e-2,1e-1,0, 0.1, 1],
                    "reg_lambda"       : [1e-5, 1e-4,1e-3,1e-2,0.1, 1, 10]}
        xgboost = Prediction()
        xgb_reg = xgboost.xgb_tune(X_train = X_train,y_train = y_train,params =params )
        xgb_reg = xgboost.xgb_train(reg_xgb = xgb_reg,X_train =  X_train,y_train =y_train )
        xgboost.xgb_pred(original_df =df_labeled ,X_test =X_test,y_test =y_test ,X_all =  X_all,reg_xgb = xgb_reg)
        return (process, xgboost)
    



features = ['Star_Ratings','VC Dummy','Internet dummy', 'firm_age',
            'top_tier_uw','perc_price_above','ASVI','mean_SVI',
            'week_-8','week_-7','week_-6','week_-5','week_-4','week_-3','week_-2','week_-1']
outcome = args.outcome
seed = args.seed
model = args.model
# process_xgbclf, prediction_xgbclf = pipeline_clf(outcome = outcome, features =features , model = 'xgboost' ,seed = seed)
process, predict = pipeline_clf(outcome = outcome, features =features , model = model ,seed = seed)


with open(f'/Users/nianyun/Documents/Pictet_IPO_Case/Result/Model/clf_{model}_{seed}.pkl','wb') as f:
    pickle.dump(predict, f)

dict_m = {}
dict_m['accuracy'] = predict.accuracy
dict_m['roc_auc_score'] = predict.roc_auc_score
dict_m['FPR'] = predict.FPR
dict_m['FNR'] = predict.FNR
dict_m['F1'] = predict.F_1

metric_df = pd.DataFrame.from_dict(dict_m , orient = 'index')
metric_df.to_csv(f'/Users/nianyun/Documents/Pictet_IPO_Case/Result/Metric/clf_{model}_{seed}_metric.csv', index = True)


# prediction_xgbclf.accuracy
# prediction_xgbclf.roc_auc_score
# prediction_xgbclf.FPR
# prediction_xgbclf.FNR
# prediction_xgbclf.F_1

# df = pd.read_csv('/Users/nianyun/Documents/Pictet_IPO_Case/Data_clean/Final_Train/IPO_train.csv')
# df['high_return'] = 0
# df.loc[df['1st_Day_Return'] > df['1st_Day_Return'].mean() , 'high_return'] = 1

# process = Preprocess()
# process.pre_process(df = df,outcome = outcome,features = features,test_size =0.2, seed = seed)
# X_train = process.X_train
# y_train = process.y_train
# # (X_test.index == y_test.index).all()
# X_test  = process.X_test
# y_test  = process.y_test
# X_all = process.X_all




# #
# predict = Prediction()

# params = {"learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
#                     "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
#                     "min_child_weight" : [ 1, 3, 5, 7 ,10],
#                     "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4,1,1.5],
#                     "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7, 0.9,1 ],
#                     "n_estimators"     : [10,50, 150, 200, 250, 300,350,400],
#                     # "reg_alpha"        : [1e-5, 1e-4,1e-3,1e-2,1e-1,0, 0.1, 1],
#                     "reg_lambda"       : [1e-5, 1e-4,1e-3,1e-2,0.1, 1, 10]}

# xgboost = Prediction()
# xgb_reg = xgboost.xgb_tune(X_train = X_train,y_train = y_train,params =params )
# xgb_reg = xgboost.xgb_train(reg_xgb = xgb_reg,X_train =  X_train,y_train =y_train )
# xgboost.xgb_pred(original_df =df_labeled ,X_test =X_test,y_test =y_test ,X_all =  X_all,reg_xgb = xgb_reg)

# model = xgboost.model
# ind = y_test.index
# model.predict_proba(X_all)[:,1]
# df_labeled = process.original_df

