import joblib
import os

# BASE_PATH will be the folder where model_loader.py is located
BASE_PATH = os.path.dirname(__file__)

def load_artifacts():
    try:
        model_path = os.path.join(BASE_PATH, 'xgboost_eeg_emotion_model.joblib')
        scaler_path = os.path.join(BASE_PATH, 'scaler.joblib')
        label_encoder_path = os.path.join(BASE_PATH, 'label_encoder.joblib')
        feature_names_path = os.path.join(BASE_PATH, 'feature_names.joblib')

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        label_encoder = joblib.load(label_encoder_path)
        feature_names = joblib.load(feature_names_path)

        print("Model and other objects loaded successfully.")
        return model, scaler, label_encoder, feature_names

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        raise
    except Exception as e:
        print(f"Error occured while loading the model: {e}")
        raise

if __name__ == "__main__":
    load_artifacts()