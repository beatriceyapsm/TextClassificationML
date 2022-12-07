import streamlit as st  # pip install streamlit
import pandas as pd  # pip install pandas
import numpy as np  # pip install numpy
import sklearn
# import imblearn

import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
# from imblearn.pipeline       import Pipeline
# from imblearn.over_sampling  import RandomOverSampler


nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')


    
#Text Pre-processing
#convert to lowercase, strip and remove punctuations
def preprocess(text):
    text = text.lower() 
    text= text.strip()  
    text= re.compile('<.*?>').sub('', text) 
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  
    text = re.sub('\s+', ' ', text)  
    text = re.sub(r'\[[0-9]*\]',' ',text) 
    text=re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text) 
    text = re.sub(r'\s+',' ',text) 
    return text

 
# STOPWORD REMOVAL
def stopword(string):
    a= [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)

#LEMMATIZATION
# Initialize the lemmatizer
wl = WordNetLemmatizer()
 
# This is a helper function to map NTLK position tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
# Tokenize the sentence
def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string)) # Get position tags
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    return " ".join(a)  

def finalpreprocess(string):
    return lemmatizer(stopword(preprocess(string)))

#Get files from user
uploaded_file1 = st.file_uploader('Upload your train data', type='xlsx', key='1')
uploaded_file2 = st.file_uploader('Upload the data you want to classify', type='xlsx', key='2')
if uploaded_file1 and uploaded_file2:
    st.markdown('---')
    df = pd.read_excel(uploaded_file1, engine='openpyxl')
    dw = pd.read_excel(uploaded_file2, engine='openpyxl')
    
    df = df.dropna() #drop rows with missing values
    df = df.set_axis(['Text', 'Class'], axis=1, copy=False) #rename columns
    dw = dw.set_axis(['Text'], axis=1, copy=False) #rename columns

    df['Text'] = df['Text'].apply(lambda x: finalpreprocess(x))
    st.dataframe(df)



    X_train, X_test, y_train, y_test = train_test_split(df['Text'],df['Class'], test_size=0.2) #split data into train and test sets


    #Build a vectorizer / classifier pipeline that filters out tokens that are too rare or too frequent
    svm_pipe1 = Pipeline([('vect',    CountVectorizer()),
                      ('tfidf',   TfidfTransformer()),
                      ('SVC',   LinearSVC(class_weight='balanced', random_state=50))])


    # #Build a grid search to tune C. 
    # #Fit the pipeline on the training set using grid search for the parameters
    param_grid = {'SVC__C':np.arange(0.01,100,10)} 
    grid_search = GridSearchCV(svm_pipe1, param_grid, cv=5, refit = True, verbose = 3)
    grid_search.fit(X_train, y_train)

    bestlinearSVC = grid_search.best_estimator_
    bestlinearSVC.fit(X_train,y_train)

    #Predict the outcome on the testing set and store it in a variable named y_predicted
    y_predicted = bestlinearSVC.predict(X_test)
    accuracy_for_test_keys = np.mean(y_predicted == y_test)
    # st.write("SVM Model Accuracy = {} %".format(accuracy_for_test_keys*100))
    st.write(classification_report(y_test, y_predicted))

    #Predict the outcome on the new set and store it in a variable named w_predicted
    dw['clean_Text'] = dw['Text'].apply(lambda x: finalpreprocess(x))
    w_predicted = bestlinearSVC.predict(dw['clean_Text'])
    dw['Predicted'] = w_predicted
    dw=dw.drop(['clean_Text'], axis=1)
    st.dataframe(dw)

