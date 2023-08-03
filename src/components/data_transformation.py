import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer   # ColumnTransformer is basically used to create pipeline
from sklearn.impute import SimpleImputer    # for missing value
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
            
    def get_data_transformer_object(self):
        """ This function is responsible for different transformation """
        try :
            numerical_columns = ["writing_score","reading_score"]
            categorical_columns = ["gender","race_ethnicity","parental_level_of_education",
                                   "lunch","test_preparation_course"]
           # gender,race_ethnicity,parental_level_of_education,lunch,test_preparation_course,math_score,reading_score,writing_score
            # Create a pipeline and run on training dataset
            num_pipeline = Pipeline(
               steps= [("imputer",SimpleImputer(strategy="median")), # Handling the missing values
                       ("scalar",StandardScaler())   # Chaning the range of numerical data
                   
               ]
            )
            
            categorical_pipeline=Pipeline(
                steps = [("imputer",SimpleImputer(strategy="most_frequent")), # Handling the missing values in categorical data
                         ("one_hot_encoder",OneHotEncoder()),
                         ("scalar",StandardScaler(with_mean=False))
                         
                ]
                
            )
            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns encoding completed")
            
            # ColumnsTransformer is basically used for combining numerical and categorical data
            preprocessor=ColumnTransformer(
                [("num_pipline",num_pipeline,numerical_columns),
                 ("cat_pipeline",categorical_pipeline,categorical_columns)]
            )
            
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
            
    
    def initiate_data_transformation(self, train_path,test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("Reading train and test data completed")
            
            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj=self.get_data_transformer_object()
            
            target_column_name="math_score"
            numerical_columns = ["writing_score","reading_score"]
           # print(train_df.columns)
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
           # print(input_feature_train_df.columns)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying preprocessing object on training and testing DataFrame")
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
        
            logging.info("Saved preprocessing objects.")
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return (train_arr,test_arr)
        
        except Exception as e:
            raise CustomException(e,sys)
       