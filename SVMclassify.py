from sklearn.datasets import load_files
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import re
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve




def main():
    d = pd.read_csv('C:/Users/91949/Desktop/ML/grpproj/1_2000_Labelled.csv')
    x= d['Comments']; y=d ['Target']
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)
    Xtrain,Xtest = preproc(xtrain,xtest)
    preds=classify(Xtrain,ytrain,Xtest,ytest)
    print(classification_report(ytest, preds))
    print(confusion_matrix(ytest, preds))
    
   
    

def preproc(xtrain,xtest):
    xtrain_1=[]
    xtest_1 =[]
    
    stemmer = PorterStemmer()
    for X in xtrain:
        Y=str(X).replace('\n','')
        
        X=WhitespaceTokenizer().tokenize(str(Y))
        X=re.sub(r'[^\w]', ' ', str(X))
        X=word_tokenize(str(X))
        stems = [stemmer.stem(token) for token in X]
        xtrain_1.append(str(stems))
    
    for X in xtest:
        Y=str(X).replace('\n','')
        
        X=WhitespaceTokenizer().tokenize(str(Y))
        X=re.sub(r'[^\w]', ' ', str(X))
        X=word_tokenize(str(X))
        stems = [stemmer.stem(token) for token in X]
        xtest_1.append(str(stems))
    
        
        
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.2)
    Xtrain = vectorizer.fit_transform(xtrain_1)
    Xtest = vectorizer.transform(xtest_1)
    return (Xtrain,Xtest)

def classify(Xtrain,ytrain,Xtest,ytest):
    model = LinearSVC(C=1000)
    model.fit(Xtrain, ytrain)
    preds = model.predict(Xtest)
    print(cross_val_score(model, Xtest, ytest, cv=2,scoring='roc_auc'))
    
    fpr, tpr, _ = roc_curve(ytest,model.decision_function(Xtest))
    plt.plot(fpr,tpr)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.plot([0, 1], [0, 1], color="green",linestyle="--")
    plt.show()
    return preds



if __name__=="__main__":
    main()


