import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import os

from src.utills import *

@dataclass
# class DataTransformationConfig:
#     preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.dill")

class DataTransformation:
    def __init__(self):
        pass
    #     self.data_transformation_config=DataTransformationConfig()
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")


            train_df.drop(columns=['religion'],axis=1,inplace=True)
            test_df.drop(columns=['religion'],axis=1,inplace=True)

            logging.info("Dropping colomn Religion")

            target_column_name="Condition"

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            train_arr = np.c_[
                input_feature_train_df, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_df, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object and final train test dataset.")

            # save_object(

            #     file_path=self.data_transformation_config.preprocessor_obj_file_path
            # )

            return (
                train_arr,
                test_arr
                # self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)