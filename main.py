# -*- coding: utf-8 -*-
"""
Created on Mon May 15 09:32:03 2017

@author: fay
"""


import util
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier





if __name__ == "__main__":
    #读取数据
    train_data = util.loadData("./data/train.csv")
    test_data = util.loadData("./data/test.csv")
    user_data = util.loadData("./data/user.csv")
    user_insapp_data = util.loadData("./data/user_installedapps.csv")
    user_app_act_data = util.loadData("./data/user_app_actions.csv")
    app_categories_data  = util.loadData("./data/app_categories.csv")
    ad_data = util.loadData("./data/ad.csv")
    postion_data = util.loadData("./data/position.csv")
    
    #构造训练特征
    train_data = pd.merge(train_data, user_data, how='left', on='userID')
    train_data = pd.merge(train_data, postion_data, how='left', on='positionID')
    train_data = pd.merge(train_data, ad_data, how='left', on='creativeID')
    ##train_data = pd.merge(train_data, )
    
    #构造测试特征
    test_data = pd.merge(test_data, user_data, how='left', on='userID')
    test_data = pd.merge(test_data, postion_data, how='left', on='positionID')
    test_data = pd.merge(test_data, ad_data, how='left', on='creativeID')
    
    #异常值处理   
    train_data = train_data.fillna(-1)
    test_data = test_data.fillna(-1)
    
    #不平衡处理
    Oversampling = train_data.loc[train_data.label ==1]
    for i in range(50):
        train_data.append(Oversampling)
    
    
    target = 'label'
    predictors = [x for x in train_data.columns if x not in [target,'conversionTime','instanceID']]
    
    clf = GradientBoostingClassifier(n_estimators=200,random_state=2017)
    clf=clf.fit(train_data[predictors],train_data[target])
    
    result = clf.predict_proba(test_data[predictors])