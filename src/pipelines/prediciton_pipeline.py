import os,sys
from src.exception import CustomException
from src.logger import logging
from src.utils import load_obj
import pandas as pd
import numpy as np

from dataclasses import dataclass

@dataclass
class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            logging.info("Initiate Prediciton Pipeline.")
            model_path = os.path.join('artifacts','mdoel.pkl')
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            
            preprocessor = load_obj(preprocessor_path)
            model = load_obj(model_path)
            logging.info("Preprocessor and Model Object Loaded Successfully.")

            features_preprocess = preprocessor.trasform(features)
            predicted_data = model.predict(features_preprocess)

            logging.info("Prediction Pipeline Completed Successfully.")
            return predicted_data
        
        except Exception as e:
            logging.info("Error occured in Prediction Pipeline.")
            raise CustomException(e,sys)

# Cement (component 1)(kg in a m^3 mixture),Blast Furnace Slag (component 2)(kg in a m^3 mixture),Fly Ash (component 3)(kg in a m^3 mixture),
# Water  (component 4)(kg in a m^3 mixture),Superplasticizer (component 5)(kg in a m^3 mixture),
# Coarse Aggregate  (component 6)(kg in a m^3 mixture),Fine Aggregate (component 7)(kg in a m^3 mixture),
# Age (day),"Concrete compressive strength(MPa, megapascals) "


class CustomData:

    def __init__(self,
                Cement:float,
                Blast_Furnace_Slag:float,
                Fly_Ash:float,
                Water:float,
                Superplasticizer:float,
                Coarse_Aggregate:float,
                Fine_Aggregate:float,
                Age:float,
                Concrete_Compressive_Strength:float    
                ):
        self.Cement = Cement
        self.Blast_Furnace_Slag = Blast_Furnace_Slag
        self.Fly_Ash = Fly_Ash
        self.Water = Water
        self.Superplasticizer = Superplasticizer
        self.Coarse_Aggregate = Coarse_Aggregate
        self.Fine_Aggregate = Fine_Aggregate
        self.Age = Age
        self.Concrete_Compressive_Strength = Concrete_Compressive_Strength
    

    def get_data_as_data_frame(self):
        try:
            logging.info("Initiate DataFrame Gathering Process.")
            custom_data_input = {
                'Cement (component 1)(kg in a m^3 mixture)'                 : [self.Cement],
                'Blast Furnace Slag (component 2)(kg in a m^3 mixture)'     : [self.Blast_Furnace_Slag],
                'Fly Ash (component 3)(kg in a m^3 mixture)'                : [self.Fly_Ash],
                'Water  (component 4)(kg in a m^3 mixture)'                 : [self.Water],
                'Superplasticizer (component 5)(kg in a m^3 mixture)'       : [self.Superplasticizer],
                'Coarse Aggregate  (component 6)(kg in a m^3 mixture)'      : [self.Coarse_Aggregate],
                'Fine Aggregate (component 7)(kg in a m^3 mixture)'         : [self.Fine_Aggregate],
                'Age (day)'                                                 : [self.Age],
                'Concrete compressive strength(MPa, megapascals) '          : [self.Concrete_Compressive_Strength]
            }

            data_input_as_data_frame = pd.DataFrame(custom_data_input)

            logging.info("DataFrame Gathered Successfully.")

            return data_input_as_data_frame
        
        except Exception as e:
            logging.info("Error occured in DaatFrame Gathering Process.")
            raise CustomException(e, sys)







