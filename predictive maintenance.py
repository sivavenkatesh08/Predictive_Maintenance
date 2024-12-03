import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC

# Title of the app
st.title('Predictive Maintenance Model')

# Upload CSV file
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())

    # Data wrangling and preprocessing steps
    rename_columns = {
        'UDI': 'ID',
        'Air temperature [K]': 'Air_temperature',
        'Process temperature [K]': 'Process_temperature',
        'Rotational speed [rpm]': 'Rotational_speed',
        'Torque [Nm]': 'Torque',
        'Tool wear [min]': 'Tool_wear',
        'Product ID': 'Product_ID',
        'Failure Type': 'Failure_type'
    }
    df.rename(rename_columns, axis=1, inplace=True)

    # Removing unnecessary columns
    drop_columns = ["ID", "Product_ID", "Target"]
    df.drop(drop_columns, axis=1, inplace=True)

    # Numeric and categorical features
    NUMERIC_FEATURES = ['Air_temperature', 'Process_temperature', 'Rotational_speed', 'Torque', 'Tool_wear']
    CATEGORIC_FEATURES = ['Type']

    # Preprocessing
    num_pipeline = Pipeline([
        ('num_features', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('cat_features', OneHotEncoder())
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num_trans', num_pipeline, NUMERIC_FEATURES),
        ('cat_trans', cat_pipeline, CATEGORIC_FEATURES)
    ])

    # Splitting data
    X = df[NUMERIC_FEATURES + CATEGORIC_FEATURES]
    y = df['Failure_type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)

    # Model pipeline
    pip_model_no_pca = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', GradientBoostingClassifier(random_state=2023))
    ])

    # Fit pipeline with sample weights
    weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)
    pip_model_no_pca.fit(X_train, y_train, model__sample_weight=weights)

    # Generate Predictions
    y_pred = pip_model_no_pca.predict(X_test)

    # Evaluation metrics
    def get_metrics(y_true, y_pred):
        f1_scores_per_class = f1_score(y_true, y_pred, average=None)
        return {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
            'Macro Recall': recall_score(y_true, y_pred, average='macro'),
            'Macro Precision': precision_score(y_true, y_pred, average='macro'),
            'Macro F1': f1_score(y_true, y_pred, average='macro'),
            'F1 Scores per Class': f1_scores_per_class
        }

    metrics = get_metrics(y_test, y_pred)

    # Display Metrics
    st.write("Model Evaluation Metrics:")
    st.json(metrics)

    # Generate and plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(y_test),
                yticklabels=np.unique(y_pred))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    st.pyplot(fig)

    # User input for prediction
    st.sidebar.header('Predict Input')
    air_temperature = st.sidebar.number_input('Air Temperature', value=0.0)
    process_temperature = st.sidebar.number_input('Process Temperature', value=0.0)
    rotational_speed = st.sidebar.number_input('Rotational Speed', value=0.0)
    torque = st.sidebar.number_input('Torque', value=0.0)
    tool_wear = st.sidebar.number_input('Tool Wear', value=0.0)
    type = st.sidebar.selectbox('Type', df['Type'].unique())

    input_data = pd.DataFrame({
        'Air_temperature': [air_temperature],
        'Process_temperature': [process_temperature],
        'Rotational_speed': [rotational_speed],
        'Torque': [torque],
        'Tool_wear': [tool_wear],
        'Type': [type]
    })

    # Make prediction
    if st.sidebar.button('Predict'):
        prediction = pip_model_no_pca.predict(input_data)
        st.write(f"Prediction: {prediction[0]}")

    # Save the model
    if st.button("Save Model"):
        with open('model.pkl', 'wb') as model_file:
            pickle.dump(pip_model_no_pca, model_file)
        st.write("Model saved successfully!")