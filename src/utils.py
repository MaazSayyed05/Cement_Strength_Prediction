import pickle
import os,sys

from src.exception import CustomException
from src.logger import logging

from sklearn.linear_model import  LinearRegression,ElasticNet,Lasso,Ridge
from sklearn.tree import  DecisionTreeRegressor
from sklearn.ensemble import  RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import  mean_squared_error,mean_absolute_error,r2_score


def save_obj(file_path,obj):
    try:
        logging.info("Initiate Object Saving Process.")
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)

        logging.info("Object Saving Process Completed Successfully.")
    
    except Exception as e:
        logging.info("Error occured in Object Saving Process.")
        raise CustomException(e, sys)

def evaluate_models(X_train,X_test,y_train,y_test,models):
    try:
        logging.info("Initiate Evaluation of Models.")
        r2_score_list = []
        mse_list = []
        mae_list = []

        for i in range(len(list(models.keys()))):
            model = list(models.values())[i]

            model.fit(X_train,y_train)

            y_pred = model.predict(X_test)

            r2_score_list.append(r2_score(y_true=y_test,y_pred=y_pred))
            mse_list.append(mean_squared_error(y_true=y_test,y_pred=y_pred))
            mae_list.append(mean_absolute_error(y_pred=y_pred,y_true=y_test))

        logging.info("Evaluation of models Completed Successfully.")
        
        return (
            r2_score_list,
            mse_list,
            mae_list
        )
    except Exception as e:
        logging.info("Error occured in Evaluaiton of Models.")
        raise CustomException(e, sys)

def load_obj(file_path):
    try:
        logging.info("Initiate Loading of Object.")
        with open(file_path,"rb") as obj:
            return pickle.load(obj)
        logging.info("Object Loading process Completed Successfully.")
    
    except Exception as e:
        logging.info("Error occured in Loading of Object.")
        raise CustomException(e, sys)




