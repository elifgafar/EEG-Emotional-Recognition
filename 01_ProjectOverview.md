# 🧠 EEG Emotion Recognition Project

This project develops a machine learning pipeline to classify emotional states (POSITIVE, NEUTRAL, NEGATIVE) based on EEG signal features.

---

## DataSet

- **Name:** EEG Brainwave Dataset: Feeling Emotions (Kaggle)  
- **Source(s):**  
  - https://www.researchgate.net/publication/329403546_Mental_Emotional_Sentiment_Classification_with_an_EEG-based_Brain-machine_Interface
  - https://www.researchgate.net/publication/335173767_A_Deep_Evolutionary_Approach_to_Bioinspired_Classifier_Optimisation_for_Brain-Machine_Interaction  
- **Samples:** 2549 rows  
- **Target Column:** `Label`  
- **Features:** EEG channels such as FP1, FP2, AF3, etc.

---

## Tools & Technologies

- Python 3.10  
- numpy  
- pandas  
- seaborn  
- matplotlib  
- scikit-learn  
- xgboost  
- joblib  
- fastapi  
- pydantic  

---

## Workflow Summary

### Data Exploration
- Checked for missing values  
- Analyzed class distribution  
- Visualized EEG feature distributions  

### Preprocessing
- Standardized features using `StandardScaler`  
- Split dataset into training -80%- and testing -20%- sets  

---

### Modeling & Evaluation

#### 1️⃣ Logistic Regression (Baseline)
- Multinomial Logistic Regression (`lbfgs`, `max_iter=1000`) 
- **Accuracy:** 62.9%

**Classification Report Highlights:**
- **NEGATIVE** → F1-score: 0.77
- **NEUTRAL** → F1-score: 0.73
- **POSITIVE** → F1-score: 0.09 (very low recall of 0.05)

- **Conclusion:** Performance was limited, a more complex model was needed

#### 2️⃣ XGBoost Classifier
- Tuned hyperparameters for optimal performance  
- **Accuracy:** 99.53%  

**Confusion Matrix:**
[[142 0 0]
[ 0 143 0]
[ 2 0 140]]

**Classification Report:**
- **NEGATIVE** → Precision: 0.99, Recall: 1.00, F1-score: 0.99  
- **NEUTRAL** → Precision: 1.00, Recall: 1.00, F1-score: 1.00  
- **POSITIVE** → Precision: 1.00, Recall: 0.99, F1-score: 0.99  

The initial model, Logistic Regression, yielded limited performance with ~63% accuracy.
In contrast, the XGBoost classifier achieved remarkable results with 99.5% accuracy and strong, balanced performance across all emotion classes.

---

## ML Pipeline

         +------------------+
         |   EEG Dataset    |
         +------------------+
                  |
                  v
     +---------------------------+
     |  Preprocessing (Scaling)  |
     +---------------------------+
                  |
                  v
     +---------------------------+
     |     Train-Test Split      |
     +---------------------------+
                  |
                  v
     +---------------------------+
     |      XGBoost Model        |
     +---------------------------+
                  |
        +-------------------+
        | Model Evaluation  |
        +-------------------+
                  |
        +--------------------+
        |  Save Model (.pkl) | 
        +--------------------+
                  |
                  v
        +---------------------+
        |  FastAPI Inference  |
        +---------------------+
                  |
                  v
      +-------------------------+
      |  JSON: Emotion Output   |
      +-------------------------+

---

## Learnings & Future Work

- XGBoost significantly outperformed the baseline Logistic Regression model  
- Achieved high accuracy and balanced F1-scores across all emotional classes  
- Strong performance confirmed via precision, recall, and confusion matrix  
- Model is suitable for practical application in EEG-based emotion detection  

**Next Steps -Sprint 2- :**
  - Feature selection or dimensionality reduction (e.g. PCA)
  - Rebalancing strategies (e.g. SMOTE)
  - Investigate time-domain or frequency-domain EEG features

---

## API Performance

The API's response time and concurrency performance were evaluated using ApacheBench.
See [05_APIPerformance.md](./APIPerformance.md) for detailed benchmark results and interactive testing guide via Swagger UI.

---

## Author👩🏽‍💻

**Elif Gafar**  
-  [LinkedIn](https://www.linkedin.com/in/elifgafar/)  
-  [Medium](https://medium.com/@elifgafar)  
- 💻 Developed the entire ML pipeline and deployed the model via FastAPI.

---

## 📜 License

MIT License – Feel free to use, modify, and share with attribution🤖

---
Stay Generative✨