from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from model_loader import load_artifacts

app = FastAPI()

# load the model and auxiliary objects
try:
    model, scaler, label_encoder, feature_names = load_artifacts()
except Exception as e:
    raise RuntimeError(f"Model and required files failed to load: {e}")

# Pydantic model for input data
class EEGFeaturesInput(BaseModel):
    features: Dict[str, Optional[float]]  #flexible, but we will control it internally

@app.get("/")
def read_root():
    return {"info": "EEG Emotion Recognition API is running and ready for predictions!"}

@app.post("/predict")
async def predict(data: EEGFeaturesInput):
    try:
        # get features in the correct order
        input_data_ordered = [data.features.get(name) for name in feature_names]
        
        # check for missing features
        if None in input_data_ordered:
            missing_features = [name for name, val in zip(feature_names, input_data_ordered) if val is None]
            raise HTTPException(
                status_code=400,
                detail=f"The following features are missing: {', '.join(missing_features)}. "
                       f"Please include all {len(feature_names)} features to get a prediction."
            )
        
        # prepare for the model (e.g scaling)
        input_df = pd.DataFrame([input_data_ordered], columns=feature_names)
        scaled_features = scaler.transform(input_df)
        
        # Model Prediction
        prediction = model.predict(scaled_features)
        
        # decode the prediction as a label
        predicted_emotion = label_encoder.inverse_transform(prediction)[0]
        
        return {"predicted_emotion": predicted_emotion}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred during prediction. Please check your input data and try again. Error details: {e}"
        )