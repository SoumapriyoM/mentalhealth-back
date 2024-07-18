from pydantic import BaseModel
import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utills import load_object
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            logging.info(f"Loading model")
            model_path=os.path.join("artifacts","model.dll")
            # preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            # print("Before Loading")
            model=load_object(file_path=model_path)
            # preprocessor=load_object(file_path=preprocessor_path)
            # print("After Loading")
            logging.info(f"Loaded model")
            print(features)
            # data_scaled=preprocessor.transform(features)
            preds=model.predict(features)
            logging.info(f"prediction done")
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 Q3A: int,
                 Q5A: int,
                 Q10A: int,
                 Q13A: int,
                 Q16A: int,
                 Q17A: int,
                 Q24A: int,
                 Q26A: int,
                 Q31A: int,
                 Q42A: int,
                 Extraverted_enthusiastic: int,
                 Critical_quarrelsome: int,
                 Dependable_self_disciplined: int,
                 Anxious_easily_upset: int,
                 Open_to_new_experiences_complex: int,
                 Reserved_quiet: int,
                 Disorganized_careless: int,
                 Calm_emotionally_stable: int,
                 Conventional_uncreative: int,
                 education: str,
                 orientation: str,
                 married: str,
                 age_group: int):

        self.Q3A = Q3A
        self.Q5A = Q5A
        self.Q10A = Q10A
        self.Q13A = Q13A
        self.Q16A = Q16A
        self.Q17A = Q17A
        self.Q24A = Q24A
        self.Q26A = Q26A
        self.Q31A = Q31A
        self.Q42A = Q42A
        self.Extraverted_enthusiastic = Extraverted_enthusiastic
        self.Critical_quarrelsome = Critical_quarrelsome
        self.Dependable_self_disciplined = Dependable_self_disciplined
        self.Anxious_easily_upset = Anxious_easily_upset
        self.Open_to_new_experiences_complex = Open_to_new_experiences_complex
        self.Reserved_quiet = Reserved_quiet
        self.Disorganized_careless = Disorganized_careless
        self.Calm_emotionally_stable = Calm_emotionally_stable
        self.Conventional_uncreative = Conventional_uncreative
        self.education = education
        self.orientation = orientation
        self.married = married
        self.age_group = age_group

    def preprocess_input(self):

        # Map the input values
        self.education = self.map_education_to_code(self.education)
        self.orientation = self.map_orientation_to_code(self.orientation)
        self.married = self.map_married_to_code(self.married)
        self.age_group = self.label_age(self.age_group)
    
    def map_education_to_code(self, education_str):
        if education_str == "Less than high school":
            return 1
        elif education_str == "High school":
            return 2
        elif education_str == "University degree":
            return 3
        elif education_str == "Graduate degree":
            return 4
        else:
            raise ValueError("Invalid education level")

    def label_age(self,age_group):
        if age_group < 20:
            return 1
        elif age_group < 25:
            return 2
        elif age_group < 30:
            return 3
        elif age_group < 35:
            return 4
        elif age_group < 40:
            return 5
        elif age_group < 50:
            return 6
        elif age_group < 60:
            return 7
        else:
            return 8
    def map_orientation_to_code(self, orientation_str):
        # Example implementation, add more mappings as needed
        if orientation_str == "Heterosexual":
            return 1
        elif orientation_str == "Bisexual":
            return 2
        elif orientation_str == "Homosexual":
            return 3
        elif orientation_str == "Asexual":
            return 4
        elif orientation_str == "Other":
            return 5
        else:
            raise ValueError("Invalid orientation")
        
    def map_married_to_code(self, married_str):
        # Example implementation, add more mappings as needed
        if married_str == "Never married":
            return 1
        elif married_str == "Currently married":
            return 2
        elif married_str == "Previously married":
            return 3
        else:
            raise ValueError("Invalid marital status")

    def get_data_as_data_frame(self):
        try:
            self.preprocess_input()
            logging.info("Data Preprocessed")
            custom_data_input_dict = {
                "Q3A": [self.Q3A],
                "Q5A": [self.Q5A],
                "Q10A": [self.Q10A],
                "Q13A": [self.Q13A],
                "Q16A": [self.Q16A],
                "Q17A": [self.Q17A],
                "Q24A": [self.Q24A],
                "Q26A": [self.Q26A],
                "Q31A": [self.Q31A],
                "Q42A": [self.Q42A],
                "Extraverted_enthusiastic": [self.Extraverted_enthusiastic],
                "Critical_quarrelsome": [self.Critical_quarrelsome],
                "Dependable_self_disciplined": [self.Dependable_self_disciplined],
                "Anxious_easily_upset": [self.Anxious_easily_upset],
                "Open_to_new_experiences_complex": [self.Open_to_new_experiences_complex],
                "Reserved_quiet": [self.Reserved_quiet],
                "Disorganized_careless": [self.Disorganized_careless],
                "Calm_emotionally_stable": [self.Calm_emotionally_stable],
                "Conventional_uncreative": [self.Conventional_uncreative],
                "education": [self.education],
                "orientation": [self.orientation],
                "married": [self.married],
                "age_group": [self.age_group],
            }
            logging.info("Data Loaded")
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            logging.error(f"Error in get_data_as_data_frame: {e}")