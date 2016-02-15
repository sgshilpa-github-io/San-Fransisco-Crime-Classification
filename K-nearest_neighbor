from datetime import datetime
import pandas as pd
from numpy import array
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import precision_score,accuracy_score,recall_score
from sklearn.neighbors import KNeighborsClassifier
from IPython.display import Image
from StringIO import StringIO
import pydot

Time=[]
Hour=[]
Minute=[]
Month=[]
Year=[]
Date=[]
Seasons=[]
WeekDays=[]
TimeSlots=[]
Category=[]
Features=[]
Latitude=[]
Longitude=[]


def parse_time(x):
    DD=datetime.strptime(x,"%Y-%m-%d %H:%M:%S")
    hours=DD.hour#*60+DD.minute
    minutes=DD.minute
    date=DD.day
    month=DD.month
    year=DD.year
    if date in range(1,11):
        time="first"
    elif date in range(11,21):
        time="second"
    else:
        time="third"
    return hours,minutes,date,month,year,time

df_train=pd.DataFrame.from_csv("train.csv",index_col=False,parse_dates=True)
df_test=pd.DataFrame.from_csv("test.csv",index_col=False,parse_dates=True)

def parse_weekday(x):
    Weekdays=["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
    for i,value in enumerate(Weekdays):
        if value==x:
            return i


#x=hours form date column
def timeslots(x):
    if x in range(5,12):
        return 0#"Morning"
    elif x in range(12,17):
        return 1#"Noon"
    elif x in range(17,23):
        return 2#"Evening"
    else:
        return 3#"Night"


 # x=month from date column
def parse_seasons(x):
    if x in [12,1,2]:
        return "Winter"
    elif x in range(3,6):
        return "Spring"
    elif x in range(6,9):
        return "Summer"
    elif x in range(9,12):
        return "Fall"

def DecisionTreeClassifier(TrainData):
    features=['Month','Date','Year']
    season=['Fall','Spring','Summer','Winter']
    district=['BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION','NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']
    days=['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday','Wednesday']
    time=['first','second','third']
    features2 = [x for x in range(0,24)]
    Minute=[x for x in range(100,160)]
    latitude=[x for x in range(948,964)]
    longitude=[x for x in range(2070,2083)]
    features=district+Minute+features2+season+time

    train,validation= train_test_split(TrainData, test_size=0.4)

    knn = KNeighborsClassifier()
    knn.fit(train[features],train['Category'])
    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',metric_params=None, n_jobs=1, n_neighbors=5, p=2,weights='uniform',multilabel=True)
    predicted=np.array(knn.predict_proba(validation[features]))
    model=knn.predict(validation[features])
    model1=knn.predict(train[features])

    print "Precision is ",precision_score(validation['Category'].values.tolist(),model,average='macro')
    print "Recall is ",recall_score(validation['Category'].values.tolist(),model,average='macro')
    print "Accuracy is ", accuracy_score(validation['Category'].values.tolist(),model)
    print "Training Accuracy is ", accuracy_score(train['Category'].values.tolist(),model1)


    result=pd.DataFrame(predicted, columns=le_crime.classes_)
    result['Predicted']=model
    result.to_csv('knnProbabilities.csv', index = True, index_label = 'Id' )

count=0
for allrows in df_train.values:
    myhour,myminute,mydate,mymonth,myyear,mytime = parse_time(allrows[0])

    Hour.append(myhour)
    Minute.append(myminute+100)
    Date.append(mydate)
    Month.append(mymonth)
    Time.append(mytime)
    Year.append(myyear)
    weekday=parse_weekday(allrows[3])
    WeekDays.append(weekday)
    Seasons.append(parse_seasons(mymonth))
    TimeSlots.append(timeslots(myhour))
    if int(allrows[7])==-120:
        Latitude.append(948)
    else:
        Latitude.append(int(((allrows[7]+122)*100)+1000))
    if int(allrows[8])==90:
        Longitude.append(2082)
    else:
        Longitude.append(int(((allrows[8]-37)*100)+2000))
    count=count+1

le_crime = preprocessing.LabelEncoder()
crime = le_crime.fit_transform(df_train['Category'])
district = pd.get_dummies(df_train['PdDistrict'])
days = pd.get_dummies(df_train['DayOfWeek'])
Seasons=pd.get_dummies(Seasons)
Hour=pd.get_dummies(Hour)
Minute=pd.get_dummies(Minute)
Latitude=pd.get_dummies(Latitude)
Longitude=pd.get_dummies(Longitude)
Time=pd.get_dummies(Time)
train_data=pd.concat([district,days,Seasons,Hour,Minute,Longitude,Latitude,Time],axis=1)
train_data['Month']=Month
train_data['Date']=Date
train_data['Year']=Year
train_data['Category']=crime

print "K nearest neighbors is done"
DecisionTreeClassifier(train_data)
