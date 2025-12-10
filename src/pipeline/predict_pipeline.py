import pandas as pd
import numpy as np
import os
import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

from src.components.data_transformation import DataTransformation

class PredictPipeline:
    def __init__(self):
        try:
            preprocessor_path = os.path.join("artifacts", "vectorizer.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            self.preprocessor = load_object(preprocessor_path)
            self.model = load_object(model_path)

            self.dt = DataTransformation()
        except Exception as e:
            raise CustomException(e,sys)
        
    def Predict(self, features):
        try:
            if not isinstance(features, (list, tuple)):
                features = [features]
            
            cleaned_features = [self.dt._clean_text(str(t) if t is not None else '' ) for t in features]

            data_scaled = self.preprocessor.transform(cleaned_features)
            preds = self.model.predict(data_scaled)

            return preds
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self, text:str):
        self.text = text

    def get_fea_as_list(self):
        try:
            return self.text
        except Exception as e:
            raise CustomException(e, sys)