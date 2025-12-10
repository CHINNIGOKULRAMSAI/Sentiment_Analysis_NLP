import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import nltk
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

nltk.download('stopwords')
nltk.download('wordnet')

@dataclass
class DataTransformationConfig:
    vectorizer_path: str = os.path.join("artifacts", "vectorizer.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.stopwords = set(stopwords.words('english'))
        self.Lemmatizer = WordNetLemmatizer()
        self.tf_idf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=5,
            max_df=0.9,
            max_features=20000,
            sublinear_tf=True,
        )

    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ''
        text = text.lower()
        text = BeautifulSoup(text, 'html.parser').get_text()
        text = re.sub(r'[^a-z0-9 ]+', ' ', text)
        tokens = text.split()
        tokens = [self.Lemmatizer.lemmatize(w) for w in tokens if w not in self.stopwords]
        return ' '.join(tokens)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Initiating data transformation")
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_df['reviewText'] = train_df['reviewText'].apply(self._clean_text)
            test_df['reviewText'] = test_df['reviewText'].apply(self._clean_text)
            logging.info("Text Cleaning is Completed")

            train_df = train_df[train_df['reviewText'].str.strip() != '']
            test_df = test_df[test_df['reviewText'].str.strip() != '']

            X_train_text = train_df['reviewText']
            X_train_target = train_df['label']

            X_test_text = test_df['reviewText']
            X_test_target = test_df['label']

            logging.info("Vectorizing text with TF-IDF.")

            X_train_vec = self.tf_idf.fit_transform(X_train_text)
            X_test_vec = self.tf_idf.transform(X_test_text)

            y_train = np.array(X_train_target).reshape(-1, 1)
            y_test = np.array(X_test_target).reshape(-1, 1)
            train_arr = hstack([X_train_vec, y_train])
            test_arr = hstack([X_test_vec, y_test])

            save_object(
                file_path=self.data_transformation_config.vectorizer_path,
                obj=self.tf_idf
            )

            return (
                train_arr,
                test_arr,
            )
        except Exception as e:
            raise CustomException(e,sys)
        