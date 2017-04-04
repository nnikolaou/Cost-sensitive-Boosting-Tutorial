# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 11:24:52 2017

@author: Nikolaos Nikolaou & Gavin Brown
"""
###############################################################################
# This code provides a template implementation of the Calibrated-AdaMEC
# method proposed by Nikolaou et al 2016, "Cost Sensitive Boosting
# Algorithms: do we really need them?".
# 
# This follows the pseudocode laid out on p15 of the supplementary
# material.  If you make use of this code, please cite the main paper.
#
# Thanks! Happy Boosting!
# Nikolaou et al.
#
###############################################################################

## Import packages
import numpy as np
import scipy.io as sio
from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import train_test_split # Instead of above package, use this with sklearn v. 0.18 and newer
from sklearn import tree#If another sklearn classifier is used, remember to import it. Ignore warning, weak learner is called with eval()
from sklearn.ensemble import AdaBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
      

def trainCalibratedAdaMEC(base_estimator, algorithm, n_estimators, calibration_method, cal_set_prcnt, X_train, y_train):
    """Train a Calibrated AdaBoost ensemble. 
    
    Parameters:
        
    base_estimator: object, sklearn supported classifier, base learner to be used
                    e.g. decision stump: 'tree.DecisionTreeClassifier(max_depth=1)'
                    remember to import the relevant packages if other base learner
                    is used
    
    algorithm: string, possible values: {'SAMME', 'SAMME.R'} AdaBoost algorithm:
               "SAMME" for discrete AdaBoost, "SAMME.R" for real AdaBoost
        
    n_estimators: integer,  AdaBoost ensemble (maximum) size 
    
    calibration_method: string, possible values: {'isotonic', 'sigmoid'} 
                        AdaBoost ensemble score calibration method (isotonic
                        regression or Platt scaling (logistic calibration), resp.)
                        
    cal_set_prcnt: float in (0, 1), size of calibration set as a percentage of the
                   training dataset size
    
    X_train: array-like, shape (n_samples, n_features), training data
    
    y_train: array-like, shape (n_samples,), training labels
    
    Returns: 
        
    AdaBoostCal: object, a calibrated adaboost classifier
    """
    #First, reserve part of the training data for calibration
    X_train, X_cal, y_train, y_cal = train_test_split(X_train, y_train, test_size=cal_set_prcnt)

    #Train an AdaBoost ensemble
    AdaBoostUncal = AdaBoostClassifier(base_estimator, algorithm=algorithm, n_estimators=n_estimators)
    AdaBoostUncal = AdaBoostUncal.fit(X_train, y_train)
	
    #Now calibrate the ensemble on the data reserved for calibration
    #cv="prefit" means that the model is already fitted and only needs calibration
    AdaBoostCal = CalibratedClassifierCV(AdaBoostUncal, cv="prefit", method=calibration_method)
    AdaBoostCal.fit(X_cal, y_cal)

    return AdaBoostCal

	
def predictCalibratedAdaMEC(CalibratedAdaBoostClassifier, threshold, X_test):
    """Output AdaMEC (AdaBoost with shifted decision threshold) predictions

          Parameters:

          CalibratedAdaBoostClassifier: object, a calibrated classifier object as
                                        returned by trainCalibratedAdaMEC()

          threshold: float in (0, 1), the classification threshold to be compared with
		             the probability estimate in order to classify an example to the
					 positive class. For minimum risk classification, it should be set
					 equal to the skew (overall asymmetry due to both class and cost
					 imbalance),i.e. C_FP*Neg / (C_FN*Pos + C_FP*Neg), where Pos and Neg
					 is the number of positive and negative examples, respectively, of
					 the training set
                     Note: We chose to have this as an argument, as the user might want to
					 adjust it to account for a change in costs and/or prior probability shift
						  
          X_test: array-like, shape (n_samples, n_features), test data

          Returns:

          y_pred_CalibratedAdaMEC: array-like, shape (n_samples), predicted classes on
                         training data

          scores_CalibratedAdaMEC: array-like, shape (n_samples), predicted scores (i.e.
                                   calibrated probability estimates) for the positive
                                   class for the training data 
    
     """    
    scores_CalibratedAdaMEC = CalibratedAdaBoostClassifier.predict_proba(X_test)[:,1]#Positive Class scores
	
    y_pred_CalibratedAdaMEC = np.zeros(X_test.shape[0])
    y_pred_CalibratedAdaMEC[np.where(scores_CalibratedAdaMEC > threshold)] = 1#Classifications, AdaMEC uses a shifted decision threshold (skew-sensitive) 

    return (y_pred_CalibratedAdaMEC, scores_CalibratedAdaMEC)