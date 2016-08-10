import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import gc
from sklearn import cross_validation
import xgboost as xgb

os.chdir("/home/vitidn/mydata/repo_git/PredictionRedHat/")

peoples = pd.read_csv("people.csv")

#no missing values for people
peoples.isnull().sum()
#inspect column's type
peoples.dtypes

#remove char_1 b/c it is duplication of char_2(???)
peoples = peoples.drop('char_1',1)

#group_1 > turn group_1 with value = 1 to 'group_other'
group_names = peoples.group_1.value_counts()
group_names = group_names[group_names == 1].index.values
peoples.loc[peoples.group_1.isin(group_names),"group_1"] = "group_other"

#convert 'date' to datetime
peoples['date'] = pd.to_datetime(peoples.loc[:,"date"])

#plot distributions of columns left
inspect_columns = list(peoples.select_dtypes(include = ['bool','O']).columns)
inspect_columns.remove("people_id")
inspect_columns.remove("group_1")
#for column in inspect_columns:
#    sns.countplot(x=column,data=peoples)
#    plt.show()

#change string columns -> factor -> numeric
inspect_columns = list(peoples.select_dtypes(include = ['O']).columns)
inspect_columns.remove("people_id")
for column in inspect_columns:
    peoples[column] = pd.Categorical(peoples[column]).codes
    
#change bool columns -> factor -> numeric
inspect_columns = list(peoples.select_dtypes(include = ['bool']).columns)
for column in inspect_columns:
    peoples[column] = peoples[column].astype('int8')
    
#load train & test data
act_train = pd.read_csv("act_train.csv")
act_test = pd.read_csv("act_test.csv")
act_test["outcome"] = -1
act_train = act_train.append(act_test,ignore_index=True)
del act_test

#convert 'date' to datetime
act_train['date'] = pd.to_datetime(act_train.loc[:,"date"])

#substring activity_id
#I guess no need for this...
#act_train['activity_id_short'] = [x[0:4] for x in act_train['activity_id'] ]

#plot distributions of string columns left 
inspect_columns = list(act_train.select_dtypes(include = ['O']).columns)
inspect_columns.remove("people_id")
#for column in inspect_columns:
#    sns.countplot(x=column,data=act_train)
#    plt.show()
    
#plot distributions of 'outcome' columns
sns.countplot(x="outcome",data=act_train)

#char_10 > turn data that has unique char_10 to 'other'
group_names = act_train.char_10.value_counts()
group_names = group_names[group_names == 1].index.values
act_train.loc[act_train.char_10.isin(group_names),"char_10"] = "other"

#replace missing value
act_train = act_train.fillna("na")

#change string columns -> factor -> numeric
inspect_columns.remove('activity_id')
for column in inspect_columns:
    act_train[column] = pd.Categorical(act_train[column]).codes

#drop duplicated data
#(no duplicate now if date is not dropped)
#act_train = act_train.drop_duplicates()

#join peoples and act_train
df = peoples.merge(act_train,on="people_id")
    
#clear memory
del peoples,act_train
gc.collect()

#calculate diffdate from peoples and act_train
df['date_diff'] = (df['date_y'] - df['date_x']).dt.days

#define train set,validate set,test set
test_set = df[df.outcome == -1]

df = df[df.outcome != -1]

##select people_id to be in training set and validate set
np.random.seed(100)
people_ids = df['people_id'].unique()

people_ids_train = np.random.choice(people_ids,size = (int)(len(people_ids) * 0.5 ),replace = False)

train_df = df[df['people_id'].isin( people_ids_train )]
validate_df = df[~(df['people_id'].isin( people_ids_train ))]

del df
gc.collect()

train_target = train_df['outcome']
train_df = train_df.drop('outcome',1)

validate_target = validate_df['outcome']
validate_df = validate_df.drop('outcome',1)

train_df = train_df.drop(['people_id','date_x','activity_id','date_y' ],1)
validate_df = validate_df.drop(['people_id','date_x','activity_id','date_y' ],1)

#construct DMatrix data
dtrain = xgb.DMatrix(train_df.as_matrix(),label = train_target)
dvalidate = xgb.DMatrix(validate_df.as_matrix(),label = validate_target)

#save dtrain,dtest,test_set for later use
dtrain.save_binary('dtrain.txt')
dvalidate.save_binary('dvalidate.txt')
test_set.to_csv("test_set.csv",index = False)

gc.collect()
#load from files
dtrain = xgb.DMatrix('dtrain.txt')
dvalidate = xgb.DMatrix('dvalidate.txt')
test_set = pd.read_csv("test_set.csv")
#CV tuning(later...)

#fit the model
param = {'booster':'gbtree','objective':'binary:logistic','eval_metric':'auc','eta':0.05,'subsample':0.75,'colsample_bytree':0.75,'min_child_weight':0,'max_depth':11}
evallist = [(dvalidate,'validate')]

xgb_model = xgb.train(param,dtrain,50,evallist,early_stopping_rounds = 10)

#save model for later use
xgb_model.save_model("xgb.model")

#load xgb_model
xgb_model = xgb.Booster({'nthread':4})
xgb_model.load_model("xgb.model")

#make a prediction on test data
test_set = test_set.drop('outcome',1)
dtest = xgb.DMatrix(test_set.drop(['people_id','date_x','activity_id','date_y' ],1).as_matrix())
predictions = xgb_model.predict(dtest)
test_set['outcome'] = predictions

#save to submit file
test_set[["activity_id","outcome"]].to_csv("submit.csv",index = False)