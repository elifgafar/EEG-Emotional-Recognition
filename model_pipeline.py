# libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import json
from sklearn.model_selection import cross_val_score
import joblib


# load the data
df = pd.read_csv("DataSet/emotions.csv")

# Exploratory Data Analysis — EDA
df.head()
df.tail()

df.rename(columns={df.columns[0]: df.columns[0].replace('# ', '')}, inplace=True)

df.shape
df.info()
df['label'].value_counts()  #khow many different emotion classes are there. how are they distributed?
df.isnull().sum().sum()  # are there any missing values in the data?
df.describe().T.head()  # a general statistical overview

# separate the "label" column as the Target variable, and the rest as features
X = df.drop('label', axis=1)
y = df['label']

# map categorical labels to numerical values, and see the resulting encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y) # y_encoded will now contain numerical values like 0, 1, 2
print(le.classes_)
print(y_encoded)

# split 80% for Training, 20% for Test data
# preserve "class proportions" in training and test sets with stratify=y_encoded, this is very important!!
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# fit "only" on the training data, and apply transform to both training and test data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# define the XGBoost classifier
xgb_model = xgb.XGBClassifier(objective='multi:softmax', 
                              num_class=len(np.unique(y_encoded)), # the number of classes will be determined automatically
                              eval_metric='mlogloss', 
                              use_label_encoder=False, 
                              random_state=42)

# train the model on the Training data
xgb_model.fit(X_train_scaled, y_train)

# predictions on the Test set
y_pred = xgb_model.predict(X_test_scaled)

# evaluate the performance
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# re-define the XGBoost classifier with consistent parameters for cross-validation on the training set
xgb_model_cv = xgb.XGBClassifier(objective='multi:softmax', 
                                 num_class=len(np.unique(y_encoded)), 
                                 eval_metric='mlogloss', 
                                 use_label_encoder=False, 
                                 random_state=42)


# perform 5-fold cv on the training data using accuracy as the scoring metric
cv_scores = cross_val_score(xgb_model_cv, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"Çapraz Doğrulama Doğruluk Skorları: {cv_scores}")
print(f"Ortalama Çapraz Doğrulama Doğruluğu: {np.mean(cv_scores):.4f}")
print(f"Çapraz Doğrulama Doğruluğu Standart Sapması: {np.std(cv_scores):.4f}")

# define the filename for the saved model
model_filename = 'xgboost_eeg_emotion_model.joblib'

# xgb_model is previously trained model object
joblib.dump(xgb_model, model_filename)
print(f"Model successfully saved as '{model_filename}'.")

# load the model
loaded_model = joblib.load(model_filename)
print(f"Model successfully loaded from '{model_filename}'.")

# predictions on the test set with the loaded model
y_pred_loaded = loaded_model.predict(X_test_scaled)

# compare performance (should be the same as the original model)
print("\n--- Loaded Model Performance ---")
print(f"Accuracy (Loaded Model): {accuracy_score(y_test, y_pred_loaded):.4f}")
print("\nConfusion Matrix (Loaded Model):\n", confusion_matrix(y_test, y_pred_loaded))
print("\nClassification Report (Loaded Model):\n", classification_report(y_test, y_pred_loaded, target_names=le.classes_))

# get Feature Importance from the XGBoost model
feature_importances = xgb_model.get_booster().get_score(importance_type='gain')

# The 'importance_type' parameter allows selecting different metrics:
# "weight": number of times a feature is used -default-
# "gain": average gain provided by each feature split -often preferred-
# "cover": number of instances covered by each feature split


# convert the dictionary to a DataFrame and sort it
feature_importance_df = pd.DataFrame(list(feature_importances.items()), columns=['Feature', 'Importance'])
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# select the top "n" most important features (e.g. top 20)
top_n_features = 20
top_features = feature_importance_df.head(top_n_features)

print(f"\nTop {top_n_features} most important features:")
print(top_features)

# save the scaler and the label encoder
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(le, 'label_encoder.joblib')

feature_names = X.columns.to_list()
joblib.dump(feature_names, 'feature_names.joblib')
print("feature_names.joblib file created.")

# save in the same file
joblib.dump(feature_names, 'feature_names.joblib')

# DIFFERENT VALUES COME WITH EACH RUN
# select a random row and convert it to a dictionary
sample_features_dict = X.sample(n=1).iloc[0].to_dict()

# convert to FastAPI format
api_request_body = {"features": sample_features_dict}

# save to file
with open("sample_input.json", "w") as f:
    json.dump(api_request_body, f, indent=4)

print("sample_input.json file successfully saved.")