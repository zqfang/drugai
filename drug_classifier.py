#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys,os
from collections import defaultdict
import sklearn ##后期添加
from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold,GridSearchCV,train_test_split
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RandomizedLogisticRegression
from boruta import BorutaPy 
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler,scale,Imputer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc,roc_auc_score
from xgboost.sklearn import XGBClassifier
sys.path.append('/home/fangzq/miniconda/envs/vg/lib/python3.6/site-packages/')#修改
import shogun
from sklearn.externals import joblib
import time


time_start=time.time()

##0.3路径设置
Result_dir = "/Users/syang/Downloads/tmp/Part1"##输出结果路径。修改
dir1="/Users/syang/Downloads/tmp"##输入数据路径。修改
if os.path.exists(Result_dir)==False:##若目标目录不存在，创建该目录
    os.mkdir(Result_dir)
if os.path.exists(Result_dir+'/SnSp')==False:##若目标目录不存在，创建该目录
    os.mkdir(Result_dir+'/SnSp')

##1 预设函数
##1.1
def nested_dict(n, type):##
    if n == 1:
        return defaultdict(type)
    else:
        return defaultdict(lambda: nested_dict(n-1, type))

##1.2 预处理PSI及TPM数据
def Pre_process(psi_train,tpm_train):##
    
    psi_DropNA = psi_train.dropna(thresh=int(Na_DropPer*psi_train.shape[1]))
    ##处理缺失值，某个基因对应的值中缺失值占比超过Na_Droper（80%）则删除该特征。该步骤移除约70%的特征。
#    psi_DropNA_fill = psi_DropNA.apply(lambda x: x.fillna(x.mean()),axis=1)##用某特征对应数值的均值填充缺失值。
    psi_DropNA_fill = psi_DropNA.apply(lambda x: x.fillna(0),axis=1)##用0填充缺失值。
    psi_DropNA_fill_DropZeroOne = psi_DropNA_fill[psi_DropNA_fill.apply(lambda x: ((x==0).mean()< ZeroOne_DropPer)&((x==1).mean()< ZeroOne_DropPer),axis=1)]
    ##处理0、1值，某个基因对应的值中缺失值占比超过ZeroOne_DropPer（80%）则删除该特征。该步骤移除约70%的特征。
    
    tpm_DropLowExp = tpm_train[tpm_train.mean(axis=1) > 0.1]
    ##移除TPM均值过小（小于0.1）的特征，该步骤约移除70%的特征。
    tpm_DropLowExp_DropZero = tpm_DropLowExp[tpm_DropLowExp.apply(lambda x: (x==0).mean()< ZeroOne_DropPer,axis=1)]
    ##移除过多（80%）TPM为0的特征，该步骤约移除不足10%的特征。
    
    Feat = pd.concat([psi_DropNA_fill_DropZeroOne, tpm_DropLowExp_DropZero], axis=0).T
    ##拼接两类特征
    
    return(Feat)

##1.3 特征排序
def Feature_sort(Feat_scale,Label,threads=4):##通过三种特征选择方法对特征进行排序
    
    ranks = {}
    ## Univariate feature selection
    Selector = SelectKBest(f_classif, k='all')
    Selector.fit_transform(Feat_scale, Label)
    ranks["Univariate_f"] = np.argsort(Selector.pvalues_)
    
    ## RandomizedLogistic regression n_jobs=**s, more robust result from bigger n_resampling
    ##从第1900左右起，后续的特征排序得较为可疑。
    rlogreg = RandomizedLogisticRegression(n_jobs=1, n_resampling=2000, selection_threshold=0, verbose=False, random_state=0)
    ##DeprecationWarning: Class RandomizedLogisticRegression is deprecated; The class RandomizedLogisticRegression is deprecated in 0.19 and will be removed in 0.21.
    ##warnings.warn(msg, category=DeprecationWarning)
    rlogreg.fit(Feat_scale, Label)
    ranks["Randomized_Logistic_f"] = np.argsort(-abs(rlogreg.scores_))
    
    ## boruta based on randomforest n_jobs=**
    rf = RandomForestClassifier(random_state=0,n_jobs=threads,max_features='auto') 
    feat_selector = BorutaPy(rf, n_estimators='auto',perc=80,random_state=0)
    feat_selector.fit(Feat_scale, Label)
    ranks["Boruta_f"] = np.argsort(feat_selector.ranking_)
    
    return(ranks)

##1.4 SVM
def SVM_training(Training,label,SVM_param,cv,threads=6):## 分类方法：SVM
     
    SVM_pipe = Pipeline([('imr',Imputer(missing_values='NaN',strategy='mean',axis=0)),
                    ('scl',StandardScaler()),
                    ('svc',SVC(kernel="linear",random_state= 1,probability=True))])
    
    SVM_gs = GridSearchCV(estimator=SVM_pipe,
                 param_grid=SVM_param,
                 scoring='roc_auc',
                 cv=cv,
                 n_jobs=threads)

    SVM_gs.fit(Training,label)
     
    SVM_pipe.set_params(**SVM_gs.best_params_)

    SVM_pipe.fit(Training,label)
    
    return(SVM_pipe)

##1.5 随机森林
def RandomForest_training(Training,label,RandomForest_param,cv,threads=6):## 分类方法：随机森林

    RandomForest_pipe = Pipeline([('imr',Imputer(missing_values='NaN',strategy='mean',axis=0)),
                              ('rfc',RandomForestClassifier())])
    
    RandomForest_gs = GridSearchCV(estimator=RandomForest_pipe,
                 param_grid=RandomForest_param,
                 scoring='roc_auc',
                 cv=cv,
                 n_jobs=threads)

    RandomForest_gs.fit(Training,label)
   
    RandomForest_pipe.set_params(**RandomForest_gs.best_params_)
    
    RandomForest_pipe.fit(Training,label)
    
    return(RandomForest_pipe)

##1.6 XGBoost
def XGBoost_training(Training,label,XGBoost_param,cv,threads=6):## 分类方法：XGBoost

    XGBoost_gs = GridSearchCV(estimator = XGBClassifier(subsample=0.8,colsample_bytree=0.8), param_grid = XGBoost_param, scoring='roc_auc',n_jobs=threads,iid=False, cv=cv)
    XGBoost_gs.fit(Training,label)
    
    bst = XGBClassifier()
    
    bst.set_params(**XGBoost_gs.best_params_)
    bst.fit(Training,label)
    
    return(bst)

##1.7 多核学习
def custom_kernel_compute(clinical_vector,max_value=100,min_value=0):## 分类方法：多核学习
    kernel_matrix = np.zeros([len(clinical_vector),len(clinical_vector)],dtype='float64')
    for i in range(len(clinical_vector)):
        for j in range(len(clinical_vector)):
            kernel_matrix[i,j] = ((max_value-min_value)-abs(clinical_vector[i]-clinical_vector[j]))/(max_value-min_value)

    return kernel_matrix

class mkl_classifier():

    def __init__(self,mkl_epsilon=0.00001,epsilon=0.0001,gaussian_width=1,poly_degree=2, svm_c = 0.01, mkl_c = 1,svm_norm = 1, mkl_norm = 1,kernel_dict={},custom_kernel_dict={}):
        self.svm_c = svm_c
        self.mkl_c = mkl_c
        self.mkl_norm = mkl_norm
        self.svm_norm = svm_norm
        self.gaussian_width = gaussian_width
        self.poly_degree = poly_degree
        self.kernel_dict = kernel_dict
        self.custom_kernel_dict = custom_kernel_dict
        self.mkl_epsilon = mkl_epsilon
        self.epsilon = epsilon
        
    def kernel_prepare(self):       
        kernel = shogun.CombinedKernel()
        for kernel_type in self.kernel_dict.keys():
            if kernel_type == 'GaussianKernel':
                for kernel_feature in self.kernel_dict[kernel_type].values():
                    kernel.append_kernel(shogun.GaussianKernel(self.gaussian_width))
            if kernel_type == 'PolyKernel':
                for kernel_feature in self.kernel_dict[kernel_type].values():
                    kernel.append_kernel(shogun.PolyKernel(10,self.poly_degree))
            if kernel_type == 'LinearKernel':
                for kernel_feature in self.kernel_dict[kernel_type].values():
                    kernel.append_kernel(shogun.LinearKernel())
                
        return kernel  
        
    def feature_prepare(self,X):
        features = shogun.CombinedFeatures()
        X = X.astype(np.float64)
        for kernel_type in self.kernel_dict.keys():
            for kernel_feature in self.kernel_dict[kernel_type].values():
                features.append_feature_obj(shogun.RealFeatures(X[:,kernel_feature].T))
        
        return features
        
    def fit(self,X,y,**params):
        for parameter, value in params.items():
            setattr(self, parameter, value)        
        
        self.mkl = shogun.MKLClassification()

        self.mkl.set_C(self.svm_c, self.svm_c)
        self.mkl.set_C_mkl(self.mkl_c)
        self.mkl.set_mkl_norm(self.mkl_norm)
        self.mkl.set_mkl_block_norm(self.svm_norm)
        self.mkl.set_mkl_epsilon(self.mkl_epsilon)
        self.mkl.set_epsilon(self.epsilon)
        
        self.kernel =  self.kernel_prepare()
        self.feats_train = self.feature_prepare(X)  
        self.kernel.init(self.feats_train, self.feats_train)
        
        if self.custom_kernel_dict:
            print("Compute custom kernel!")
            for kernel_feature in self.custom_kernel_dict.values():
                custom_kernel_matrix = custom_kernel_compute(X[:,kernel_feature])
                CustomKernel = shogun.CustomKernel()
                CustomKernel.set_full_kernel_matrix_from_full(custom_kernel_matrix)
                self.kernel.append_kernel(CustomKernel)   
                del CustomKernel
        
        self.mkl.set_kernel(self.kernel)
        y = y.astype(np.float64)
        self.mkl.set_labels(shogun.BinaryLabels(y))
        self.mkl.train()
        self.kernel_weights = self.kernel.get_subkernel_weights()

    def predict(self, X,**params):
        for parameter, value in params.items():
            setattr(self, parameter, value)  
        
        self.feats_test = self.feature_prepare(X)  
        self.kernel.init(self.feats_train, self.feats_test)
        self.mkl.set_kernel(self.kernel)
        binary = self.mkl.apply_binary()
        
        return binary.get_labels()
    
    def predict_proba(self, X,**params):# 该函数可计算样本是正例的可能性，作为最终的临床诊断建议         
        
        for parameter, value in params.items():
            setattr(self, parameter, value)  
        
        self.feats_test = self.feature_prepare(X)  
        self.kernel.init(self.feats_train, self.feats_test)
        self.mkl.set_kernel(self.kernel)
        binary = self.mkl.apply_binary()
        binary.scores_to_probabilities()
        
        v1 = binary.get_values()
        v0 = 1 - v1
        proba = np.concatenate((v0.reshape(len(v0),1),(v1.reshape(len(v1),1))), axis=1)
        
        return proba
    
    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)

        return self

    def get_params(self, deep=False):

        return {param: getattr(self, param) for param in dir(self) if not param.startswith('__') and not callable(getattr(self,param))}

    def score(self, X,y,**params):
        for parameter, value in params.items():
            setattr(self, parameter, value) 

        predicted = self.predict_proba(X)[:,1]
        
        return roc_auc_score(y,predicted)  ##Default:GridSearchCV scoring is roc_auc_score
    
def MKL_training(Training,label,MKL_param,cv,threads=6):

    feature_name = Training.columns.tolist()
    label=label.astype(np.float64).values
    
    train_psi = [feature_name.index(x) for x in feature_name if x.startswith("ENSG")]
    train_exp = [feature_name.index(x) for x in feature_name if not x.startswith("ENSG")]
    kernel_dict = {'GaussianKernel':{'1':train_psi,'2':train_exp},'PolyKernel':{'1':train_psi,'2':train_exp},'LinearKernel':{'1':train_psi,'2':train_exp}}
    
    MKL_pipe = Pipeline([('imr',Imputer(missing_values='NaN',strategy='mean',axis=0)),
                         ('scl',StandardScaler()),  
                         ('mkl',mkl_classifier(kernel_dict=kernel_dict))])
    
    MKL_gs = GridSearchCV(estimator=MKL_pipe,
                 param_grid=MKL_param,
                 cv=cv,
                 n_jobs=threads)

    MKL_gs.fit(Training,label)
   
    MKL_pipe.set_params(**MKL_gs.best_params_)
    
    MKL_pipe.fit(Training,label)
    
    return(MKL_pipe)


##1.8 SVM训练_single
def SVM_training_single(Training,label,SVM_param):
     
    SVM_pipe = Pipeline([('imr',Imputer(missing_values='NaN',strategy='mean',axis=0)),
                    ('scl',StandardScaler()),
                    ('svc',SVC(kernel="linear",random_state= 1,probability=True))])
     
    SVM_pipe.set_params(**SVM_param)

    SVM_pipe.fit(Training,label)
    
    return(SVM_pipe)

##1.8 随机森林训练_single
def RandomForest_training_single(Training,label,RandomForest_param):

    RandomForest_pipe = Pipeline([('imr',Imputer(missing_values='NaN',strategy='mean',axis=0)),
                              ('rfc',RandomForestClassifier())])
   
    RandomForest_pipe.set_params(**RandomForest_param)
    
    RandomForest_pipe.fit(Training,label)
    
    return(RandomForest_pipe)

##1.8 XGBoost训练_single
def XGBoost_training_single(Training,label,XGBoost_param):

    XGBoost_param['subsample'] = 0.8
    XGBoost_param['colsample_bytree'] = 0.8

    bst = XGBClassifier()
    bst.set_params(**XGBoost_param)
    bst.fit(Training,label)
    
    return(bst)

##1.8 MKL训练_single
def MKL_training_single(Training,label,MKL_param):

    feature_name = Training.columns.tolist()
    label=label.astype(np.float64).values
    
    train_psi = [feature_name.index(x) for x in feature_name if x.startswith("ENSG")]
    train_exp = [feature_name.index(x) for x in feature_name if not x.startswith("ENSG")]
    kernel_dict = {'GaussianKernel':{'1':train_psi,'2':train_exp},'PolyKernel':{'1':train_psi,'2':train_exp},'LinearKernel':{'1':train_psi,'2':train_exp}}
    
    MKL_pipe = Pipeline([('imr',Imputer(missing_values='NaN',strategy='mean',axis=0)),
                         ('scl',StandardScaler()),  
                         ('mkl',mkl_classifier(kernel_dict=kernel_dict))])
   
    MKL_pipe.set_params(**MKL_param)
    
    MKL_pipe.fit(Training,label)
    
    return(MKL_pipe)

##1.9 整理所有的参数组合
def parameter_combination(param_dict):##输入为算法对应的参数字典
    
    name=[]
    value =[]
    combination_num= 1
    res_dict=dict() 
    datatype = []
    
    for k in param_dict.keys():
        name.append(k)
        value.append(param_dict[k])
        datatype.append(type(param_dict[k][0]))
        combination_num=combination_num*len(param_dict[k])
    
    combination=np.array(np.meshgrid(*value)).reshape(len(name),combination_num).T
    ##meshgrid?

    for x in combination: 
        
        key = ';'.join([x+':'+str(y) for (x,y) in zip(name,x.tolist())])
        res_dict[key] =  dict(zip(name,[fn(v) for fn,v in zip(datatype,x.tolist())]))
    
    return(res_dict)

##1.10 从整理的参数组合中选出最优参数
def get_bestParams(crossvalidate):
    best_auc = 0

    for k1 in crossvalidate.keys():
        for k2,v2 in crossvalidate[k1].items():
            if best_auc < np.mean(v2):
                best_auc = np.mean(v2)
                best_param = k1+'#'+str(k2)
                
    return(best_param,best_auc)

##2 ===============================分类器=======================================
##2.1 导入样本的PSI，TPM，样本标签等文件
Lung_tpm = pd.read_csv(dir1+'/tpm.csv',index_col=0)#修改
##癌症样本的基因表达数据。
Lung_psi = pd.read_csv(dir1+'/psi.csv',index_col=0)#修改
##癌症样本的可变剪接数据。
Sample_sheet = pd.read_csv(dir1+'/sample_sheet.csv')
##标签文件，记录样本名称及分类。其中癌症样本308个，正常样本240个，良性样本74个（暂未使用）。
Sample_sheet['Label'] = Sample_sheet['Group'].replace(['Lung','NO'],[1,-1])#修改
## 标注肺癌患者和健康患者的标签为1和-1。
## ----------------------------------------------------------------------------
##2.2：用训练集建模，特征提取并根据选取的前排特征的交集以及模型的超参数组合，分别构建多个模型。
Test_SizePer = 0.3##
Test_KFold = 5
Na_DropPer = 0.8
ZeroOne_DropPer = 0.8
Validation_SizePer = 0.3# compared to Feature train set
Validation_KFold = 5
Round = 1
Top_Feat_num = list(np.arange(500,8500,500))
CV = StratifiedKFold(n_splits=3,shuffle=True,random_state=6)
SVM_param={'svc__C':[0.01,0.1,1,10,100]}
XGBoost_param = {'max_depth':[3,5,7,9],'min_child_weight':[x for x in range(1,6,2)],'gamma':[i/10.0 for i in range(0,5)],'n_estimators':[x for x in range(10,100,10)]}
RandomForest_param={'rfc__n_estimators':[x for x in range(10,100,10)],'rfc__max_depth':[3,5,7,9,11]}
MKL_param = { 'mkl__svm_c': [0.01,0.1,1],
               'mkl__mkl_c': [0.01,0.1,1],
               'mkl__svm_norm': [1,2],
               'mkl__mkl_norm':[1,1.5,2],
               'mkl__gaussian_width':[0.1,0.5,1,2,5],
               'mkl__poly_degree':[2,5,10]
               }

crossvalidate_svm = nested_dict(2, list)
crossvalidate_randomforest = nested_dict(2, list)
crossvalidate_xgboost = nested_dict(2, list)
crossvalidate_mkl = nested_dict(2, list)

trainall, test, trainall_lab, test_lab = train_test_split(Sample_sheet['SampleName'], Sample_sheet['Label'], test_size=Test_SizePer,stratify=Sample_sheet['Label'], random_state=6)
##trainall，test分别为训练集（全）、测试集的样本，trainall_lab，test_lab为对应标签    

Validation_skf2 = StratifiedShuffleSplit(n_splits=Validation_KFold,test_size=Validation_SizePer,random_state=6)
  
SVM_param_combination = parameter_combination(SVM_param)
RandomForest_param_combination = parameter_combination(RandomForest_param)
XGBoost_param_combination = parameter_combination(XGBoost_param)
MKL_param_combination = parameter_combination(MKL_param)
##对于每种方法，获取其所有的参数组合

for train, validation in Validation_skf2.split(trainall, trainall_lab):
##利用验证集在4中机器学习方法中选择最优的特征数与超参数组合    
    os.popen('mkdir '+Result_dir+'/Round'+str(Round))
    time.sleep(3)
    sample_type = pd.Series(['Test']*len(test)+['Validation']*len(validation)+['Train']*len(train),index=list(test)+trainall.iloc[validation].tolist()+trainall.iloc[train].tolist())
    ##按顺序标注样本位于哪个数据集
    sample_type.to_csv(Result_dir+'/Round'+str(Round)+'/sample_type.csv',index=False)
    
    Feature_data = Pre_process(Lung_psi[trainall.iloc[train]],Lung_tpm[trainall.iloc[train]])
    Feature_data_scale = scale(Feature_data)
    Feature_rank = Feature_sort(Feature_data_scale,trainall_lab.iloc[train])
    ##数据预处理及特征选择
    
    pd.DataFrame(Feature_rank).to_csv(Result_dir+'/Round'+str(Round)+'/feature_rank.csv',index=False)
   
    train_data = pd.concat([Lung_psi[trainall.iloc[train]],Lung_tpm[trainall.iloc[train]]],axis=0).T
    train_label = trainall_lab.iloc[train]
    validation_data = pd.concat([Lung_psi[trainall.iloc[validation]],Lung_tpm[trainall.iloc[validation]]],axis=0).T
    validation_label = trainall_lab.iloc[validation]
    validation_data.to_csv(Result_dir+'/Round'+str(Round)+'/validation_data.csv',index=False)
    train_data.to_csv(Result_dir+'/Round'+str(Round)+'/train_data.csv',index=False)
    ##获取训练及验证数据
    
    for Num in Top_Feat_num:
        
        Overlap = set(Feature_data.columns.tolist())
        for key,value in Feature_rank.items():
            
            Overlap = Overlap.intersection(Feature_data.columns[value[0:Num]])
        
        validation_data_use = validation_data[list(Overlap)]
        train_data_use = train_data[list(Overlap)]
        
        print("Round:{Round};Num:{Num};Overlap_F:{OF};".format(Round=Round,Num=Num,OF=len(Overlap)))
        ##实际使用特征最多为1600个左右
        
        ## svm 
        for k,v in SVM_param_combination.items():
            SVM_mod=SVM_training_single(train_data_use,train_label,v)
            SVM_pred = SVM_mod.predict_proba(validation_data_use)##待考证
            crossvalidate_svm[k][Num].append(roc_auc_score(validation_label, SVM_pred[:,1]))
        
        ## randomforest 
        for k,v in RandomForest_param_combination.items():
            Randomforest_mod = RandomForest_training_single(train_data_use,train_label,v)
            Randomforest_pred = Randomforest_mod.predict_proba(validation_data_use)
            crossvalidate_randomforest[k][Num].append(roc_auc_score(validation_label, Randomforest_pred[:,1]))
            
        ## xgboost
        ##耗时约5分钟
        for k,v in XGBoost_param_combination.items():
            XGBoost_mod = XGBoost_training_single(train_data_use,train_label,v)
            XGBoost_pred = XGBoost_mod.predict_proba(validation_data_use)
            crossvalidate_xgboost[k][Num].append(roc_auc_score(validation_label,XGBoost_pred[:,1]))
        
        ## mkl
        ##参数组合较多，平均每个样本耗时约4秒
        for k,v in MKL_param_combination.items():
            mkl_mod = MKL_training_single(train_data_use,train_label,v)
            mkl_pred = mkl_mod.predict_proba(validation_data_use)
            crossvalidate_mkl[k][Num].append(roc_auc_score(validation_label,mkl_pred[:,1]))
         
    Round = Round+1
#test_data = pd.concat([Lung_psi[test],Lung_tpm[test]],axis=0).T
## ----------------------------------------------------------------------------
##2.3在测试集上对两类样本的进行分类
svm_param,svm_auc = get_bestParams(crossvalidate_svm)
randomforest_param,randomforest_auc = get_bestParams(crossvalidate_randomforest)
xgboost_param,xgboost_auc = get_bestParams(crossvalidate_xgboost)
mkl_param,mkl_auc = get_bestParams(crossvalidate_mkl)
##得到最优的参数组合及最优AUC

Feature_data = Pre_process(Lung_psi[trainall],Lung_tpm[trainall])
Feature_data_scale = scale(Feature_data)
Feature_rank = Feature_sort(Feature_data_scale,trainall_lab)
##上一步在训练集中提取特征，该步骤在训练集+验证集中提取特征

trainall_data = pd.concat([Lung_psi[trainall],Lung_tpm[trainall]],axis=0).T
test_data = pd.concat([Lung_psi[test],Lung_tpm[test]],axis=0).T

##svm
param,Num = svm_param.split('#')
Overlap_svm = set(Feature_data.columns.tolist())
for key,value in Feature_rank.items():
    Overlap_svm = Overlap_svm.intersection(Feature_data.columns[value[0:int(Num)]])
svm_param_dict=SVM_param_combination[param]

SVM_mod=SVM_training_single(trainall_data[list(Overlap_svm)],trainall_lab,svm_param_dict)
SVM_pred = SVM_mod.predict_proba(test_data[list(Overlap_svm)])

##randomforest
param,Num = randomforest_param.split('#')
Overlap_randomforest = set(Feature_data.columns.tolist())
for key,value in Feature_rank.items():
    Overlap_randomforest = Overlap_randomforest.intersection(Feature_data.columns[value[0:int(Num)]])
randomforest_param_dict= RandomForest_param_combination[param]

Randomforest_mod = RandomForest_training_single(trainall_data[list(Overlap_randomforest)],trainall_lab,randomforest_param_dict)
Randomforest_pred = Randomforest_mod.predict_proba(test_data[list(Overlap_randomforest)])

##xgboost
param,Num = xgboost_param.split('#')
Overlap_xgboost = set(Feature_data.columns.tolist())
for key,value in Feature_rank.items():
    Overlap_xgboost = Overlap_xgboost.intersection(Feature_data.columns[value[0:int(Num)]])
xgboost_param_dict= XGBoost_param_combination[param]

XGBoost_mod = XGBoost_training_single(trainall_data[list(Overlap_xgboost)],trainall_lab,xgboost_param_dict)
XGBoost_pred = XGBoost_mod.predict_proba(test_data[list(Overlap_xgboost)])

##mkl
param,Num = mkl_param.split('#')
Overlap_mkl = set(Feature_data.columns.tolist())
for key,value in Feature_rank.items():
    Overlap_mkl = Overlap_mkl.intersection(Feature_data.columns[value[0:int(Num)]])
mkl_param_dict=MKL_param_combination[param]

mkl_mod = MKL_training_single(trainall_data[list(Overlap_mkl)],trainall_lab,mkl_param_dict)
mkl_pred = mkl_mod.predict_proba(test_data[list(Overlap_mkl)])
## *_pred为两列，第一列为预测得分，第二列为1-预测得分，使用第二列打分
np.savetxt(Result_dir+'/Randomforest_pred.txt',Randomforest_pred)
np.savetxt(Result_dir+'/SVM_pred.txt',SVM_pred)
np.savetxt(Result_dir+'/XGBoost_pred.txt',XGBoost_pred)
np.savetxt(Result_dir+'/mkl_pred.txt',mkl_pred)
##保存四个模型的预测得分

##3 ============================结果输出及保存===================================
##3.1 在测试集上的分类结果,ROC 
fig = plt.figure(figsize=(10,8),dpi=600)
plt.style.use("ggplot") 
colors=['green','blue','orange','red']
linestypes=['-','-','-','-']
labels = ['RandomForest','SVM','XGBoost','MKL']
Pred = [Randomforest_pred[:,1],SVM_pred[:,1],XGBoost_pred[:,1],mkl_pred[:,1]]
for p,lab,col,ls in zip(Pred,labels,colors,linestypes):
    fpr,tpr,thresholds = roc_curve(y_true= test_lab,y_score=p)
    roc_auc = auc(x=fpr,y=tpr)
    plt.plot(fpr,tpr,color=col,linestyle=ls,label='%s (auc = %0.2f)'%(lab, roc_auc),linewidth=6)
    plt.legend(loc='lower right',fontsize=20)
    
plt.plot([0,1],[0,1],linestyle='--',color='gray',linewidth=6)
plt.xlim([-0.01,1.01])
plt.xticks(fontsize=20,color='black')
plt.yticks(fontsize=20,color='black')
plt.ylim([-0.01,1.01])
plt.grid(True)  
plt.xlabel('False Positive Rate',fontsize=20,color='black')
plt.ylabel('True Positive Rate',fontsize=20,color='black')
plt.title('ROC analysis',fontsize=25)
plt.savefig(Result_dir+'/ROC.png')

##3.2 在测试集上的分类结果，PR曲线
from sklearn.metrics import precision_recall_curve,average_precision_score
fig = plt.figure(figsize=(10,8),dpi=600)
plt.style.use("ggplot") 
for p,lab,col,ls in zip(Pred,labels,colors,linestypes):
    precision, recall, _ = precision_recall_curve(test_lab,p)
    AP =  average_precision_score(test_lab, p)
    plt.plot(recall,precision,color=col,linestyle=ls,linewidth=6,label='%s (average_precision = %0.2f)'%(lab, AP))
    plt.legend(loc='lower right',fontsize=20)

plt.xlim([-0.01,1.01])
plt.xticks(fontsize=20,color='black')
plt.yticks(fontsize=20,color='black')
plt.ylim([-0.01,1.01])
plt.grid(True)  
plt.xlabel('Recall',fontsize=20,color='black')
plt.ylabel('Precision',fontsize=20,color='black')
plt.title('Precision-Recall Curve',fontsize=25)
plt.savefig(Result_dir+'/PR.png')

##3.3 保存（输出）各个模型的准确率
trans = lambda x: -1 if x < 0.5 else 1
acc_r="Randomforest_accuracy:{score}".format(score=sklearn.metrics.accuracy_score(test_lab,list(map(trans,Randomforest_pred[:,1]))))
acc_s="SVM_accuracy:{score}".format(score=sklearn.metrics.accuracy_score(test_lab,list(map(trans,SVM_pred[:,1]))))
acc_x="XGBoost_accuracy:{score}".format(score=sklearn.metrics.accuracy_score(test_lab,list(map(trans,XGBoost_pred[:,1]))))
acc_m="MKL_accuracy:{score}".format(score=sklearn.metrics.accuracy_score(test_lab,list(map(trans,mkl_pred[:,1]))))
#print("Randomforest_accuracy:{score}".format(score=sklearn.metrics.accuracy_score(test_lab,list(map(trans,Randomforest_pred[:,1])))))
#print("SVM_accuracy:{score}".format(score=sklearn.metrics.accuracy_score(test_lab,list(map(trans,SVM_pred[:,1])))))
#print("XGBoost_accuracy:{score}".format(score=sklearn.metrics.accuracy_score(test_lab,list(map(trans,XGBoost_pred[:,1])))))
#print("MKL_accuracy:{score}".format(score=sklearn.metrics.accuracy_score(test_lab,list(map(trans,mkl_pred[:,1])))))
acc=acc_r+','+acc_s+','+acc_x+','+acc_m
file=open(Result_dir+'/accuracy.txt','w') 
file.write(acc); 
file.close()
##将准确率存入accuracy.txt文件

##3.4 计算四种模型的在阈值为0.5时的Sn、Sp值
Randomforest_score=Randomforest_pred[:,1]
SVM_score=SVM_pred[:,1]
XGBoost_score=XGBoost_pred[:,1]
mkl_score=mkl_pred[:,1]
thres=0.5
test_lab1=test_lab.tolist()
##Randomforest_tfpn=[0,0,0,0,0,0]依次代表tp/fp/tn/fn/Sn/Sp
def SnSp_05(test,score):
    tfpn=[0,0,0,0,0,0]
    for n in range(len(test)):
        if score[n]>=0.5 and test[n]==1:
            tfpn[0]+=1#tp
        elif score[n]>=0.5 and test_lab1[n]==-1:
            tfpn[1]+=1#fp
        elif score[n]<0.5 and test_lab1[n]==-1:
            tfpn[2]+=1#tn
        else:
            tfpn[3]+=1#fn
    tfpn[4]=tfpn[0]/(tfpn[0]+tfpn[3])#Sn=tp/(tp+fn)
    tfpn[5]=tfpn[2]/(tfpn[2]+tfpn[1])#Sp=tn/(tn+fp)
    return(tfpn)
Randomforest_SnSp05=SnSp_05(test_lab1,Randomforest_score) 
SVM_SnSp05=SnSp_05(test_lab1,SVM_score)   
XGBoost_SnSp05=SnSp_05(test_lab1,XGBoost_score) 
mkl_SnSp05=SnSp_05(test_lab1,mkl_score)
fp=open(Result_dir+'/SnSp/SnSp05.txt','w')
fp.write('tp'+'\t'+'fp'+'\t'+'tn'+'\t'+'fn'+'\t'+'Sn'+'\t'+'Sp'+'\n')
fp.write('Randomforest'+str(Randomforest_SnSp05)+'\n')
fp.write('SVM'+str(SVM_SnSp05)+'\n')
fp.write('XGBoost'+str(XGBoost_SnSp05)+'\n')
fp.write('mkl'+str(mkl_SnSp05)+'\n')
fp.close()
    
##3.5 计算所有的Sn、Sp值
def SnSp(test,score):
    score1=sorted(score)
    tfpn=[[0 for i in range(7)] for j in range(len(test))]
    for n in range(len(test)):
        tp=0
        fp=0
        tn=0
        fn=0
        thres=score1[n]
        for m in range(len(test)):
            if score[m]>=thres and test[m]==1:
                tp+=1#tp
            elif score[m]>=thres and test_lab1[m]==-1:
                fp+=1#fp
            elif score[m]<thres and test_lab1[m]==-1:
                tn+=1#tn
            else:
                fn+=1#fn
        Sn=tp/(tp+fn)
        Sp=tn/(tn+fp)
        tfpn[n]=[thres,tp,fp,tn,fn,Sn,Sp]
    return(tfpn)
Randomforest_SnSp=SnSp(test_lab1,Randomforest_score) 
SVM_SnSp=SnSp(test_lab1,SVM_score)   
XGBoost_SnSp=SnSp(test_lab1,XGBoost_score) 
mkl_SnSp=SnSp(test_lab1,mkl_score)

fp=open(Result_dir+'/SnSp/Randomforest_SnSp.txt','w+')
fp.write('score\ttp\tfp\ttn\tfn\tSn\tSp\n')
for line in Randomforest_SnSp:
    fp.write(str(line)[1:len(str(line))-1].replace(', ','\t')+'\n')
fp.close()

fp=open(Result_dir+'/SnSp/SVM_SnSp.txt','w+')
fp.write('score\ttp\tfp\ttn\tfn\tSn\tSp\n')
for line in SVM_SnSp:
    fp.write(str(line)[1:len(str(line))-1].replace(', ','\t')+'\n')
fp.close()

fp=open(Result_dir+'/SnSp/XGBoost_SnSp.txt','w+')
fp.write('score\ttp\tfp\ttn\tfn\tSn\tSp\n')
for line in XGBoost_SnSp:
    fp.write(str(line)[1:len(str(line))-1].replace(', ','\t')+'\n')
fp.close()

fp=open(Result_dir+'/SnSp/mkl_SnSp.txt','w+')
fp.write('score\ttp\tfp\ttn\tfn\tSn\tSp\n')
for line in mkl_SnSp:
    fp.write(str(line)[1:len(str(line))-1].replace(', ','\t')+'\n')
fp.close()

##3.6 保存训练好的四种模型
joblib.dump(mkl_mod,Result_dir+'/Lung_MKL_mod.pkl')#修改
joblib.dump(Randomforest_mod,Result_dir+'/Lung_Randomforest_mod.pkl')#修改
joblib.dump(SVM_mod,Result_dir+'/Lung_SVM_mod.pkl')#修改
joblib.dump(XGBoost_mod,Result_dir+'/Lung_XGBoost_mod.pkl')#修改

##3.7 保存特征集、训练及测试数据
f = open(Result_dir+'/Feature_use_svm.txt','w')
f.write(';'.join(list(Overlap_svm)))
f.close()
f = open(Result_dir+'/Feature_use_randomforest.txt','w')
f.write(';'.join(list(Overlap_randomforest)))
f.close()
f = open(Result_dir+'/Feature_use_xgboost.txt','w')
f.write(';'.join(list(Overlap_xgboost)))
f.close()
f = open(Result_dir+'/Feature_use_mkl.txt','w')
f.write(';'.join(list(Overlap_mkl)))
f.close()

trainall_data.to_csv(Result_dir+'/trainall_data.csv',index=False)
test_data.to_csv(Result_dir+'/test_data.csv',index=False)

##3.8
#train_all = trainall_data.apply(lambda x: x.fillna(x.mean()),axis=1)##用某特征对应数值的均值填充缺失值。
train_all = trainall_data.apply(lambda x: x.fillna(0),axis=1)##用0填充缺失值。
train_all=train_all.T
feature_all=train_all.columns.values.tolist()
mean_value=[]
for n in range(len(feature_all)):
    mean_value.append(train_all[feature_all[n]].mean())
##保存train_all以及mean_value
f = open(Result_dir+'/feature_all.txt','w')
f.write(';'.join(feature_all))
f.close()
mean_value=pd.DataFrame(mean_value)
mean_value.to_csv(Result_dir+'/mean_value.csv',index=False)

##0.4 计算程序结束的时间与程序耗时
time_end=time.time()
print(time_end-time_start)
print('s')