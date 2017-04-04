# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 11:24:52 2017

@author: Nikolaos Nikolaou & Gavin Brown
"""
###############################################################################

#AdaMECvsAdaBoost() compares the standard AdaBoost algorithm with calibrated
#AdaBoost using a skew-sensitive decision threshold (Calibrated AdaMEC in the
#paper), for the purposes of skew-sensitive classification. Calibrated AdaMEC
#aims to minimize the Expected misclassification Cost (Risk)
                                                                                
###############################################################################
#Parameters:
#
#dataset: string, provided datasets: {"survival", "ionosphere", "parkinsons",
#                                     "krvskp", "liver", "pima", "musk2",
#                                     "congress_orig", "landsatM", "wdbc_orig",
#                                     "heart_orig", "sonar_orig", "spliceM",
#                                     "german_credit", "semeion_orig",
#                                     "waveformM", "spambase", "mushroom", }
#
#
#C_FP: float, cost of a false positive
#C_FN: float, cost of a false negative
#      Note: The costs above only capture one aspect of the asymmetry. Each
#            dataset has a different degree of class imbalance. The combined
#            measure of cost & class imbalance (skew) is captured by
#            C_FP_effective (or equivalently C_FN_effective)
#
#base_estimator: object, sklearn supported classifier, base learner to be used
#                e.g. decision stump: 'tree.DecisionTreeClassifier(max_depth=1)'
#                remember to import the relevant packages if other base learner
#                is used
#
#algorithm: string, possible values: {'SAMME', 'SAMME.R'} AdaBoost algorithm:
#           "SAMME" for discrete AdaBoost, "SAMME.R" for real AdaBoost
#
#n_estimators: integer,  AdaBoost ensemble (maximum) size 
#
#calibration_method: string, possible values: {'isotonic', 'sigmoid'} 
#                    AdaBoost ensemble score calibration method (isotonic
#                    regression or Platt scaling (logistic calibration), resp.)
#
#test_set_prcnt: float in (0, 1), size of test set as a percentage of the
#                original dataset size
#
#cal_set_prcnt: float in (0, 1), size of calibration set as a percentage of the
#               training dataset size
#
#
#Returns: Prints classification and probability estimation results of 
#         standard AdaBoost and calibrated AdaBoost using a skew-sensitive
#         decision threshold (Calibrated AdaMEC in the paper), when run on the
#         provided dataset and cost-setup using the specified parameters
#
###############################################################################

## Import packages
import numpy as np
import scipy.io as sio
from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import train_test_split # Instead of above package, use this with sklearn v. 0.18 and newer
from sklearn import tree #If another sklearn classifier is used, remember to import it. Ignore warning, weak learner is called with eval()
from sklearn.ensemble import AdaBoostClassifier
import CalibratedAdaMEC# Our calibrated AdaMEC training and prediction methods can be found here
from sklearn import metrics
import os 

def AdaMECvsAdaBoost(dataset, C_FP, C_FN, base_estimator, algorithm, n_estimators, calibration_method, test_set_prcnt, cal_set_prcnt):       
    ## Load data
    mat_contents = sio.loadmat(os.getcwd()+'\\Datasets\\'+dataset+'.mat')
    data = mat_contents['data']
    target = np.asarray([float(i) for i in mat_contents['labels'].ravel()])
    
    target[np.where(target != 1)] = 0 # One-vs-all if multiclass
    
    ## Split the data into training and test sets      
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_set_prcnt)  
    
    Pos = sum(y_train[np.where(y_train == 1)]) #Number of positive training examples --estimate of prior of positive class
    Neg = len(y_train) - Pos                   #Number of negative training examples --estimate of prior of negative class
    
    C_FP_effective = C_FP*Neg / (C_FN*Pos + C_FP*Neg) #Positive skew (overall importance of a single positive example)
    #C_FN_effective = 1 - C_FP_effective              #Negative skew (overall importance of a single negative example)
    
	#Define weak learner
    base_estimator = eval(base_estimator)
	
    ## Train ensembles
    #I.Train an AdaBoost ensemble (algorithm="SAMME" for discrete AdaBoost, algorithm="SAMME.R" for real AdaBoost)
    AdaBoost = AdaBoostClassifier(base_estimator, algorithm=algorithm, n_estimators=n_estimators)
    AdaBoost = AdaBoost.fit(X_train, y_train)                 
    
    #II.Train a Calibrated AdaBoost ensemble
    AdaBoostCal = CalibratedAdaMEC.trainCalibratedAdaMEC(base_estimator, algorithm, n_estimators, calibration_method, cal_set_prcnt, X_train, y_train)
    
    
    ## Generate predictions
    #I.AdaBoost predictions and scores
    scores_AdaBoost = AdaBoost.predict_proba(X_test)[:,1]#Positive Class scores
    y_pred_AdaBoost = np.zeros(X_test.shape[0])
    y_pred_AdaBoost[np.where(scores_AdaBoost > 0.5)] = 1#Classifications, the standard AdaBoost decision rule corresponds to a threshold of 0.5 (skew-insensitive) 
    
    #II.Calibrated AdaMEC predictions and scores   
    y_pred_CalibratedAdaMEC, scores_CalibratedAdaMEC = CalibratedAdaMEC.predictCalibratedAdaMEC(AdaBoostCal, C_FP_effective, X_test)
    

    ##Print results: comment/uncomment to your liking!

    # #Confusion matrices
    #print('AdaBoost Confusion Matrix:')
    #conf_mat_AdaBoost = metrics.confusion_matrix(y_test, y_pred_AdaBoost)
    #print(conf_mat_AdaBoost)
    #print('Calibrated AdaMEC Confusion Matrix:')
    #conf_mat_CalibratedAdaMEC = metrics.confusion_matrix(y_test, y_pred_CalibratedAdaMEC)
    #print(conf_mat_CalibratedAdaMEC)
    
    # #Accuracy (lower means better *skew-insensitive* classification).
    #            Note: Not a good measure for *skew-sensitive* learning.
    #print('Accuracy:')
    #print('\t\t\tAdaBoost: {0}'.format(metrics.accuracy_score(y_test, y_pred_AdaBoost)))
    #print('\t\t\tCalibrated AdaMEC: {0}'.format(metrics.accuracy_score(y_test, y_pred_CalibratedAdaMEC)))
    
     #Brier Score (lower means better probability estimates)
    print('Brier Score:')
    print('\t\t\tAdaBoost: {0}'.format(metrics.brier_score_loss(y_test, scores_AdaBoost)))
    print('\t\t\tCalibrated AdaMEC: {0}'.format(metrics.brier_score_loss(y_test, scores_CalibratedAdaMEC)))
    
     #Negative Log-likelihood (lower means better probability estimates)
    print('Negative Log-likelihood:')
    print('\t\t\tAdaBoost: {0}'.format(metrics.log_loss(y_test, scores_AdaBoost)))
    print('\t\t\tCalibrated AdaMEC: {0}'.format(metrics.log_loss(y_test, scores_CalibratedAdaMEC)))

     #Misclassification Cost (lower means better skew-sensitive classification)
    print('Misclassification Cost:')
    conf_mat_AdaBoost = metrics.confusion_matrix(y_test, y_pred_AdaBoost)#Confusion matrix  
    cost_AdaBoost = conf_mat_AdaBoost[0,1]*C_FP_effective + conf_mat_AdaBoost[1,0]*(1-C_FP_effective)#Skew-Sensitive Cost
    print('\t\t\tAdaBoost: {0}'.format(cost_AdaBoost))
    conf_mat_CalibratedAdaMEC = metrics.confusion_matrix(y_test, y_pred_CalibratedAdaMEC)#Confusion matrix
    cost_AdaMEC = conf_mat_CalibratedAdaMEC[0,1]*C_FP_effective + conf_mat_CalibratedAdaMEC[1,0]*(1-C_FP_effective)#Skew-Sensitive Cost
    print('\t\t\tCalibrated AdaMEC: {0}'.format(cost_AdaMEC))
    if cost_AdaBoost > cost_AdaMEC:
        print('Calibrated AdaMEC outperformed AdaBoost!')
    else:
        print('AdaBoost produced a lower cost solution this time. Try again.')
        print('Calibrated AdaMEC should lead to lower cost in expectation.')