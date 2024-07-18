from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline, CustomData  # Adjust the import path based on your project structure
import os
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import string

application=FastAPI()
app = application
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, you can also specify specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

class InputData(BaseModel):
    Q3A: int
    Q5A: int
    Q10A: int
    Q13A: int
    Q16A: int
    Q17A: int
    Q24A: int
    Q26A: int
    Q31A: int
    Q42A: int
    Extraverted_enthusiastic: int
    Critical_quarrelsome: int
    Dependable_self_disciplined: int
    Anxious_easily_upset: int
    Open_to_new_experiences_complex: int
    Reserved_quiet: int
    Disorganized_careless: int
    Calm_emotionally_stable: int
    Conventional_uncreative: int
    education: str
    orientation: str
    married: str
    age_group: int

# Create instance of PredictPipeline and CustomData
predict_pipeline = PredictPipeline()

@app.post("/predict/")
async def predict(data: InputData):
    try:
        # Create CustomData instance
        custom_data = CustomData(
            Q3A=data.Q3A,
            Q5A=data.Q5A,
            Q10A=data.Q10A,
            Q13A=data.Q13A,
            Q16A=data.Q16A,
            Q17A=data.Q17A,
            Q24A=data.Q24A,
            Q26A=data.Q26A,
            Q31A=data.Q31A,
            Q42A=data.Q42A,
            Extraverted_enthusiastic=data.Extraverted_enthusiastic,
            Critical_quarrelsome=data.Critical_quarrelsome,
            Dependable_self_disciplined=data.Dependable_self_disciplined,
            Anxious_easily_upset=data.Anxious_easily_upset,
            Open_to_new_experiences_complex=data.Open_to_new_experiences_complex,
            Reserved_quiet=data.Reserved_quiet,
            Disorganized_careless=data.Disorganized_careless,
            Calm_emotionally_stable=data.Calm_emotionally_stable,
            Conventional_uncreative=data.Conventional_uncreative,
            education=data.education,
            orientation=data.orientation,
            married=data.married,
            age_group=data.age_group
        )

        # Get data as DataFrame
        data_df = custom_data.get_data_as_data_frame()
        print(data_df)

        # Predict using the model
        predictions = predict_pipeline.predict(data_df)

        print(predictions)

        return predictions[0] # Convert numpy array to list

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1")
