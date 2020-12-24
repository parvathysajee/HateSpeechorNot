from hatesonar import Sonar
import json
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
 
def main():
    d = pd.read_csv('C:/Users/91949/Desktop/ML/grpproj/1_2000_Labelled.csv')
    x= d['Comments']; y=d ['Target']
    ypred=hatedetect(x)
    print(classification_report(y, ypred))
    print(confusion_matrix(y,ypred))
    

 
 
def hatedetect(hds):
    ypred=[]
    sonar = Sonar()
    for k in hds:
        check=[]
        res=sonar.ping(text=k)
        res_final=json.dumps(res)
#person_dict = json.loads(str(res))

        res_dict = json.loads(res_final)
        #print(res_dict)
#for x in res_dict["classes"]:
        if res_dict["classes"][0]["confidence"]>res_dict["classes"][1]["confidence"]:
            check=res_dict["classes"][0]["confidence"]
            val=0
        else:
            check=res_dict["classes"][1]["confidence"]
            val=1
        if check>res_dict["classes"][2]["confidence"]:
            check_final=check
        

        else:
            check_final=res_dict["classes"][2]["confidence"]
            val=2

        if val==0 or val==1:
            yp=0
        else:
            yp=1

        ypred.append(yp)
    return ypred



if __name__=="__main__":
    main()

  



