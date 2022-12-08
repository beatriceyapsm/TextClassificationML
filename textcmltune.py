import streamlit as st  # pip install streamlit
import pandas as pd  # pip install pandas
import numpy as np  # pip install numpy
import sklearn
import imblearn

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
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from imblearn.over_sampling  import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline


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

    X_train, X_test, y_train, y_test = train_test_split(df['Text'],df['Class'], test_size=0.2) #split data into train and test sets

    # fit and apply the transform
    tv = TfidfVectorizer() 
    ros = RandomOverSampler(sampling_strategy='minority')
    X_ROS, y_ROS = ros.fit_resample(tv.fit_transform(X_train), y_train)
    X_test = tv.transform(X_test)

    #Build a vectorizer / classifier pipeline that filters out tokens that are too rare or too frequent
    
    #Load Estimators
    rf=RandomForestClassifier()
    nb=MultinomialNB()
    svc=SVC()
    # rf=RandomForestClassifier(random_state=4, n_jobs=-1, max_features="sqrt", warm_start=True)
    # nb=MultinomialNB(alpha=0.01)
    # svc=SVC(random_state=4, kernel='rbf')

    ensemble_clf=[rf, nb, svc] 

    #Parameters
    paramrf={"clf__max_depth": range(10,30,10), "clf__min_samples_leaf": range(20,30,5),
         "clf__n_estimators":range(500,800,200)}
    paramnb={
    "clf__alpha": [0.01, 0.1, 1.0]
    }
    paramsvc={"clf__kernel":["rbf", "poly"], "clf__gamma": ["auto", "scale"], "clf__degree":range(1,6,1)}

    parameters_list=[paramrf, paramnb, paramsvc]
    model_log=pd.DataFrame(["_rf", "_nb", "_svc"])

    for i in range(len(ensemble_clf)): # ('tf-idf', TfidfVectorizer()),
        Grid=GridSearchCV(Pipeline([ ('clf',ensemble_clf[i])]), param_grid=parameters_list[i], 
                        n_jobs=-1, cv=3, verbose=3,scoring='average_precision' ).fit(X_ROS, y_ROS)
        model_log[i]=(ensemble_clf[i],Grid.best_score_, Grid.best_estimator_)  

    model_log= model_log.T
    model_log=model_log.sort_values(by=1, ascending=False).reset_index(drop = True)
    st.dataframe(model_log)
    
    bestmodelt=eval(model_log.loc[0][2])
    bestmodelt.fit(X_ROS, y_ROS)
    y_predicted = bestmodelt.predict(X_test)
    st.write(classification_report(y_test, y_predicted))


    #Predict the outcome on the new set and store it in a variable named w_predicted
    dw['clean_Text'] = dw['Text'].apply(lambda x: finalpreprocess(x))
    X_pred = tv.transform(dw['clean_Text'])
    w_predicted = bestmodelt.predict(X_pred)
    dw['Predicted'] = w_predicted
    dw=dw.drop(['clean_Text'], axis=1)
    st.dataframe(dw)

