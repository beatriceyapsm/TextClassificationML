import streamlit as st  # pip install streamlit
import pandas as pd  # pip install pandas
import numpy as np  # pip install numpy
import sklearn
import imblearn
# example of random oversampling to balance the class distribution
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
# define dataset
#Get files from user
uploaded_file1 = st.file_uploader('Upload your train data', type='xlsx')
if uploaded_file1 :
    st.markdown('---')
    df = pd.read_excel(uploaded_file1, engine='openpyxl')
    
    df = df.dropna() #drop rows with missing values
    df = df.set_axis(['Text', 'Class'], axis=1, copy=False) #rename columns
    X = df['Text']
    y = df['Class']
    # define oversampling strategy
    oversample = RandomOverSampler(sampling_strategy='minority')
    # fit and apply the transform
    X_over, y_over = oversample.fit_resample(X, y)
    # summarize class distribution
    st.write(len(y_over))