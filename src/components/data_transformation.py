import pandas as pd
import numpy as np
import os,sys
from src.exception import CustomException
from src.logger import logging

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression,Lasso,ElasticNet,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from dataclasses import dataclass
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Initiate Data Preprocessing Object Process.")
            preprocessor_obj = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_path,
                obj=preprocessor_obj
            )

            logging.info("Data Preprocessing Object Process Completed Successfully.")

            return preprocessor_obj
        
        except Exception as e:
            logging.info("Error Occured in Dta Preprocessing Object Process.")
            raise CustomException(e, sys)


    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            logging.info("Initiate Data Transformaiton Process.")
            train_data_set = pd.read_csv(train_data_path)
            test_data_set = pd.read_csv(test_data_path)

            logging.info("Train Dataset and Test Dataste fetched Successfully.")
            target_feature = 'Concrete compressive strength(MPa, megapascals) '

            input_feature_train_data = train_data_set.drop(target_feature,axis=1)
            target_feature_train_data = train_data_set[[target_feature]]
            logging.info("Train Dataset Segregated Successfully.")

            input_feature_test_data = test_data_set.drop(target_feature,axis=1)
            target_feature_test_data = test_data_set[[target_feature]]
            logging.info("Test Dataset Segregated Successfully.")

            preprocessor = self.get_data_transformation_object()
            logging.info("Initiate Preprocess of Dependent Features.")

            input_feature_train_data_arr = preprocessor.fit_transform(input_feature_train_data)
            input_feature_test_data_arr  = preprocessor.transform(input_feature_test_data)

            logging.info("Preprocessing of Dependent Features Completed Successfully.")

            train_arr = np.c_[input_feature_train_data_arr,np.array(target_feature_train_data)]
            test_arr  = np.c_[input_feature_test_data_arr,np.array(target_feature_test_data)]

            logging.info("Data Transformation Process Completed Successfully.")
            return (
                train_arr,
                test_arr
            )
        
        except Exception as e:
            logging.info("Error Occured in Data Transformation Process.")
            raise CustomException(e, sys)















