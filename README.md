# Predictive_Maintenance Model App

This repository contains the code for a **Predictive Maintenance Model App** built using Streamlit. The application enables users to upload a CSV file, preprocess the data, train a machine learning model, and make predictions. The app provides interactive data exploration and visualization features, as well as options for model evaluation and prediction.

***Features***


**Upload CSV Files:** Users can upload datasets in CSV format for processing.


**Data Preprocessing** 
     
      -> Rename and clean column names.
      
      
      -> Handle numeric and categorical features with pipelines.
**Model Training**
     
      
      -> Utilizes a *GradientBoostingClassifier* with preprocessing pipelines.
     
      
      -> Handles imbalanced datasets using class weights.


**Model Evaluation**

      
      -> Metrics: Accuracy, Balanced Accuracy, Recall, Precision, and F1 Score.
      
      
      -> Confusion Matrix visualization using Seaborn.


**User Input for Predictions:** Provides a sidebar form for user inputs and displays the prediction.


**Model Saving:** Option to save the trained model as a .pkl file.


***Dependencies***


The application relies on the following Python libraries:

       
       -> streamlit
       
       
       -> pandas
       
       
       -> numpy
       
       
       -> matplotlib
       
       
       -> seaborn
       
       
       -> scikit-learn
       
       
       -> imbalanced-learn



***How to Use***


**Upload Dataset:** Upload a CSV file with relevant features for predictive maintenance.


**Explore Data:** View a preview of the uploaded data.


**Train Model:** Train a *GradientBoostingClassifier* using preprocessed data.


**Evaluate Model:** 
        
        
        -> Check metrics like Accuracy, F1 Score, and more.
        
        
        -> Visualize the confusion matrix.


**Make Predictions:**
        
        
        -> Input feature values in the sidebar form.
        
        
        -> Click "Predict" to get the failure type prediction.


**Save Model:** Save the trained model for future use.


***App Workflow***


**Data Preprocessing:**
        
        
        -> Numeric features: Standard scaling.
        
        
        -> Categorical features: One-hot encoding.


**Model Training:** *GradientBoostingClassifier* with class weighting.


**Model Evaluation:** Calculate metrics and visualize the confusion matrix.


**User Predictions:**

        
        -> Accepts user inputs through the Streamlit sidebar.
        
        
        -> Displays predictions for the given input.
