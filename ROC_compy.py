from sklearn.datasets import load_files
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import matplotlib.patches as mpatches
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.naive_bayes import BernoulliNB
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier




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
   
    model1 =  LogisticRegression()
    model1.fit(Xtrain, ytrain)
    preds1 = model1.predict(Xtest)
    fpr, tpr, t= roc_curve(ytest,model1.decision_function(Xtest))
    #print(model1.decision_function(Xtest))
    plt.plot(fpr,tpr,label= "Logistic classifier",color='red')

    model2 = BernoulliNB()
    model2.fit(Xtrain, ytrain)
    preds2 = model2.predict(Xtest)
    x_1=model2.predict_proba(Xtest)
    print(x_1)
    fpr, tpr, t = roc_curve(ytest,model2.predict_proba(Xtest)[:,1])
    plt.plot(fpr,tpr,label= "Naive Bayes",color='blue')

    model3 =DecisionTreeClassifier(max_depth=1)
    model3.fit(Xtrain, ytrain)
    preds3 = model3.predict(Xtest)
    fpr, tpr, _ = roc_curve(ytest,model3.predict_proba(Xtest)[:,1])
    plt.plot(fpr,tpr,label= "Decision tree",color='black')

    model4 = KNeighborsClassifier(n_neighbors=3)
    model4.fit(Xtrain, ytrain)
    preds4 = model4.predict(Xtest)
    fpr, tpr, t = roc_curve(ytest,model4.predict_proba(Xtest)[:,1])
    plt.plot(fpr,tpr,label= "Kneighbors",color='yellow')
    
    model5 = LinearSVC(C=1000)
    model5.fit(Xtrain, ytrain)
    preds5 = model5.predict(Xtest)
    fpr, tpr,t = roc_curve(ytest,model5.decision_function(Xtest))
    plt.plot(fpr,tpr,label= "SVM",color='cyan')

    model6 = RandomForestClassifier(max_depth=2, random_state=0)
    model6.fit(Xtrain, ytrain)
    preds6 = model6.predict(Xtest)
    fpr, tpr, t = roc_curve(ytest,model6.predict_proba(Xtest)[:,1])
    plt.plot(fpr,tpr,label= "Random Forest",color='pink')
    
    
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    red_patch = mpatches.Patch(color='red', label='Logistic classifier')
    blue_patch = mpatches.Patch(color='blue', label='Naive Bayes')
    black_patch = mpatches.Patch(color='black', label='Decision tree')
    yellow_patch = mpatches.Patch(color='yellow', label='Kneighbors')
    cyan_patch = mpatches.Patch(color='cyan', label='SVM')
    pink_patch = mpatches.Patch(color='pink', label='Random Forest')

    plt.legend(handles=[red_patch, blue_patch,black_patch,yellow_patch,cyan_patch,pink_patch])

    plt.plot([0, 1], [0, 1], color="green",linestyle="--")
    plt.title("ROC curves")
    plt.show()
    return preds



if __name__=="__main__":
    main()


