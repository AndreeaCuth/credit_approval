from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI()


class CreditApplication(BaseModel):
    Gender: int              # 0 / 1 (Male = 0, Female = 1)
    Age: float
    Debt: float
    Married: int             # 0 / 1 (Yes/No)
    BankCustomer: int        # 0 / 1
    Industry: str            # keep as string (not encoded)
    Ethnicity: str           # keep as string  (not encoded, white, black, asian...)
    YearsEmployed: float
    PriorDefault: int        # 0 / 1
    Employed: int            # 0 / 1
    CreditScore: int
    DriversLicense: int      # 0 / 1 (Yes/No)
    Citizen: str             # string (ByBirth / ByOtherMeans etc.)
    ZipCode: int
    Income: float


# load models
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "model", "credit_approval_model.pkl")
label_encoders_path = os.path.join(script_dir, "model", "label_encoders.pkl")

try:
    model = joblib.load(model_path)
    label_encoders = joblib.load(label_encoders_path)
except Exception as e:
    raise RuntimeError(f"Error loading model files: {str(e)}")

# define categorical columns that need encoding
CATEGORICAL_COLS = ['Industry', 'Ethnicity', 'Citizen']


@app.post("/predict")
def predict(application: CreditApplication):
    try:
        input_df = pd.DataFrame([application.dict()])

        # creditScore normalized
        input_df["CreditScore"] = input_df["CreditScore"].clip(lower=300, upper=850)

        # encode categorical
        for col in CATEGORICAL_COLS:
            if col in input_df.columns:
                le = label_encoders.get(col)
                if le is not None:
                    valid_classes = list(le.classes_)
                    input_df[col] = input_df[col].astype(str).apply(
                        lambda x: x if x in valid_classes else valid_classes[0]
                    )
                    input_df[col] = le.transform(input_df[col])


        # ensure the correct column order for the model
        input_df = input_df[model.feature_names_in_]

        # predict
        prediction = model.predict(input_df)[0]
        result = {
            "result": "Approved" if prediction == 1 else "Rejected"
        }

        if hasattr(model, "predict_proba"): # show the probability
            result["probability"] = float(model.predict_proba(input_df)[0][1])

        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
