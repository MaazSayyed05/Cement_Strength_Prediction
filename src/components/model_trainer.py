import numpy as np
from sklearn.linear_model import  LinearRegression,ElasticNet,Lasso,Ridge
from sklearn.tree import  DecisionTreeRegressor
from sklearn.ensemble import  RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import  mean_squared_error,mean_absolute_error,r2_score

from src.exception import CustomException
from src.logger import logging

from src.utils import evaluate_models,save_obj
import os,sys
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    model_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info("Initiate Model Training Process.")

            models = {
                'Linear Regression'     : LinearRegression(),
                'Decision Tree'         : DecisionTreeRegressor(),
                'Random Forest'         : RandomForestRegressor(),
                'Support Vector'        : SVR(),
                'Lasso Regression'      : Lasso(),
                'Ridge Regression'      : Ridge(),
                'ElasticNet Regression' : ElasticNet()
            }

            X_train,X_test,y_train,y_test = train_arr[:,:-1],test_arr[:,:-1], train_arr[:,-1],test_arr[:,-1]
            logging.info("Splitting of Train and Test Dataste into Dependent and Independent features Completed Successfully.")

            score_list, mse_list, mae_list = evaluate_models(X_train,X_test,y_train,y_test,models)

            model_name_list = list(models.keys())

            model_list = list(models.values())

            best_score_index = score_list.index(max(score_list))

            best_model_score = score_list[best_score_index]

            best_model_mse = mse_list[best_score_index]

            best_model_mae = mae_list[best_score_index]

            best_model = model_list[best_score_index]

            logging.info(f"Best Model: {model_name_list[best_score_index]},  R2 Score: {best_model_score}  MSE: {best_model_mse}  MAE: {best_model_mae}.")

            save_obj(
                file_path=self.model_trainer_config.model_path, 
                obj=best_model)

            logging.info("Model Trainign Process Completed Successfully.")

        except Exception as e:
            logging.info("Error Occured in Model Trainign Process.")
            raise CustomException(e,sys)



    










