# Credit-Card-Fraud-Detection

---

## overview
This repository provides a comprehensive guide to a credit card fraud detection project based on a Kaggle dataset.
The main challenge of this project is the highly imbalanced dataset, where only 492 out of 284,807 transactions are fraudulent, representing just 0.172% of all transactions.

Another major challenge is that most features are anonymized, which makes feature engineering and interpretability more difficult.

To address the class imbalance, I experimented with various modeling techniques and sampling strategies to optimize the F1-score. After extensive testing, the best two models were scikit-learn’s MLPClassifier and Random Forest, both trained on raw, scaled data, achieving an F1-score of approximately 0.83.

Interestingly, sampling techniques did not improve performance in this project. Undersampling significantly reduced model accuracy and generalization, while oversampling provided little to no benefit.

---


## Why Are Imbalanced Datasets a Challenge in Fraud Detection?

Imbalanced datasets are very common in fraud detection problems, where fraud cases make up only a small percentage of all transactions. This creates several challenges:

- **Biased predictions:**
  If we train a model on the imbalanced data as it is, the model may learn to always predict the majority class (e.g., "not fraud"). This can     result in high accuracy but poor fraud detection, because most fraud cases will be missed.
  
- **Mismatch with real-world data:**
   Resampling techniques like oversampling or undersampling change the class distribution in the training set. For example, turning 1% fraud      cases into 35% changes the nature of the data(in real life). This can help the model learn to detect fraud better, but it no longer reflects    the true distribution seen in real-world scenarios.

- **Performance depends on the dataset:**
   While techniques like oversampling can improve results, they don’t always work well on every dataset. Their success depends on the data         characteristics and the models used.

---

## Install Requirements

To install the required dependencies, run:

```bash
pip install -r requirements.txt

```
---

## Repo structure 

```bash

│   .gitattributes
│   credit_fraud_data_utils.py
│   Credit_Fraud_Detection_Report.pdf
│   credit_fraud_val_utils.py
│   Fraud_Credit_Train.py
│   Fraud_detection_EDA.ipynb
│   predict.py
│   README.md
│   requirements.txt
│
├───.all_trained_models
│      
├───.best_models
│       model_scaled_MLPClassifier_default_thresh.pkl
│       model_scaled_RandomForest_default_thresh.pkl
│
├───.classification_reports_for_all_models
│      
├───.confusion_metrics_for_all_models
│       
├───.data
│    data_splits.zip
│        ├───train.csv
│        ├───val.csv
│        ├───test.csv
│        ├───trainval.csv
├───.f1_score_visualization(for_all_models_through_all_experiments)
│       f1_heatmap.png
│       f1_heatmap_best_threshold.png
│       f1_scores_cost_sensitive.png
│       f1_scores_cost_sensitive_scaled.png
│       f1_scores_oversample.png
│       f1_scores_oversample_scaled.png
│       f1_scores_over_and_under.png
│       f1_scores_over_and_under_scaled.png
│       f1_scores_raw.png
│       f1_scores_scaled.png
│       f1_scores_undersample.png
│       f1_scores_undersample_scaled.png

```
```.gitattributes```: Git configuration file for line endings and file attribute settings.

```credit_fraud_data_utils.py```: Contains data preprocessing utilities (scaling, sampling, splitting, loading).

```Credit_Fraud_Detection_Report.pdf```: Final project report with detailed methodology, results, and key insights.

```credit_fraud_val_utils.py```: Evaluation utilities: model validation, threshold tuning, loading models, and result saving.

```Fraud_Credit_Train.py```: Main training script to run multiple experiments with different models and sampling strategies.

```Fraud_detection_EDA.ipynb```: Exploratory Data Analysis notebook with visualizations and data exploration on training data.

```predict.py```: Inference script for loading trained models and predicting fraud on new datasets.

```README.md```: Documentation file (this file).

```requirements.txt```: Lists Python libraries and their versions required to run the project.

```.all_trained_models/```
Stores all trained models from different experiments. Useful for comparing multiple runs.

```.best_models/```
- Contains the best performing models ( MLPClassifier, Random Forest) based on F1-score.
- model_scaled_MLPClassifier_default_thresh.pkl – Best MLP model using scaled data.
- model_scaled_RandomForest_default_thresh.pkl – Best Random Forest model using scaled data.

```.classification_reports_for_all_models/```
Folder where classification reports (precision, recall, F1-score) are saved for each experiment.

```.confusion_metrics_for_all_models/```
Contains confusion matrix plots to visualize model performance across classes for each experiment.

```.data/```
Folder containing the dataset in zipped format.

- **train.csv**: Training set
- **val.csv**: Validation set
- **test.csv**: Test set
- **trainval.csv**: Combined training + validation set

```.f1_score_visualization(for_all_models_through_all_experiments)/```
Visualizations comparing F1-scores across different models and sampling techniques.
Includes:
- **f1_heatmap.png** — F1 scores for all models and experiments combined in a single figure.
- **f1_scores_*.png** — F1 scores for various experiments (raw, scaled, over/undersampling, cost-sensitive, etc.)

---
## How to run 
**1- Train Models (if you want to train from scratch)**

```bash
python train.py \
  --train_path path/to/train.csv \
  --val_path path/to/val.csv \
  --scaler standard \
  --run all
```

```--train_path``` : path to the training dataset

```--val_path```  path to the validation dataset

```--scaler```: choose between standard, minmax, robust, or none

```--run```: use all to run all experiments or best to run only the best configuration.

**2- Run predictions with a saved model (inference phase)**

```bash
python predict.py \
  --model_path "model_scaled_MLPClassifier_default_thresh.pkl" \
  --test_path "path/to/test.csv" \
  --output_dir "outputs/predictions"
```

```--model_path```: path to your saved model .pkl file

```--test_path```: path to the test dataset CSV file

```--output_dir```: directory to save predictions and metrics (optional)

---

## Models and Techniques Used to Handle Class Imbalance

In this section, I describe the techniques and machine learning models I used to address the challenge of class imbalance in fraud detection.

**1- Techniques**:
To handle the imbalance between the majority (non-fraud) and minority (fraud) classes, I applied the following strategies:

-** Using the Data As-Is**: Training the model on the original imbalanced dataset to serve as a baseline and see if the resampling techniques will improve the performance or harm it.
- **Undersampling**: Reducing the number of majority class samples to balance the dataset.
- **Oversampling**: using SMOTE (Synthetic Minority Oversampling Technique), generating synthetic examples for the minority class to increase its representation.
- **Combined Sampling**: Applying both oversampling and undersampling to create a more b  alanced dataset.
- **Cost-Sensitive Learning**: Assigning a higher cost to misclassifying fraud cases to force the model to focus more on the minority class.
  

**2- Models**:
I experimented with a variety of machine learning algorithms to evaluate how well they perform with different sampling strategies:
- Logistic Regression
- Random Forest
- XGBoost
- CatBoost
- MLP (Multi-Layer Perceptron)
- Voting Classifier


**Note**: I removed XGBoost and CatBoost because they overfitted the data and decreased the overall performance. This also helped me see how the voting classifier performs without them.

---

## Top models 
**Criteria Considered**
In real-world projects, the **best model** depends on the business’s priorities—such as agreed performance targets and acceptable error trade-offs. In this project, **my primary selection metric was the F1-score**.
Since two models achieved almost the same F1-score, I applied additional considerations :
- **Cost of Missing Fraud vs Cost of False Alarms**
- **Customer Experience (too many blocks = angry customers)**
- **Investigation Capacity (Can your team handle 1000 alerts/day?)**


**Top two models**
Two models stood out during evaluation:

- **Random Forest (raw data or raw data with scaling)**
- **MLPClassifier (scaled data)**

**Random Forest results at default threshold**
- **f1_score_val = 0.8363**
- **pr_auc_val = 0.8525**

**MLPClassifier results at default threshold**
- **f1_score_val = 0.8323**
- **pr_auc_val = 0.7798**


**Both models achieved almost the same F1 score at the default threshold. The only difference is that the Random Forest model had a higher PR-AUC. So, the choice of which model to use depends on your criteria — you can try both and select the one that performs best for your needs.**



**Now, let’s take another point of view and look at the confusion matrix.**


<p align="center">
  <img src="https://github.com/user-attachments/assets/92411390-5dcb-4ad1-9bd0-b0fc6a4b2bcf" alt="conf_matrix_scaled_MLPClassifier_default" width="45%" />
  <img src="https://github.com/user-attachments/assets/ed5ca1c6-a9a7-46c0-b76d-4de16e5fd78b" alt="conf_matrix_scaled_RandomForest_default" width="45%" />
</p>


**MLP Classifier:**                                                 **Random Forest:**                     
- True Positives (TP): 72                                           - True Positives (TP): 69
- False Positives (FP): 11                                          - False Positives (FP): 6
- False Negatives (FN): 18                                          - False Negatives (FN): 21
  


**Choose MLP if**:
- Fraud losses are extremely costly
- You can handle customer complaints from false blocks
- Investigation capacity is high


**Choose Random Forest if**:
- Customer experience is critical
- Investigation resources are limited
- False alarms are expensive to resolve


**In the end, it’s a trade-off. The decision should be made by looking at the problem from multiple perspectives, not just one.**

**Note: If you want a full analysis of all models across all experiments, go to Section 4 of the report for a detailed explanation.**

---

## Results

**1- Results for all models and experiments are based on the default threshold applied to the validation dataset.**

<img width="1200" height="700" alt="f1_heatmap" src="https://github.com/user-attachments/assets/79bb5eb4-684e-444a-8779-18166a8c9929" />

**2-Results for all models and experiments are based on the optimized threshold applied to the validation dataset.**
<img width="1200" height="700" alt="f1_heatmap_best_threshold" src="https://github.com/user-attachments/assets/3a2580ba-f252-4f7c-b787-a892bee2e1e9" />











  


