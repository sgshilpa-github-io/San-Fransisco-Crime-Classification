from datetime import datetime
import pandas as pd
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier,GradientBoostingClassifier)
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
from sklearn.metrics import precision_score,accuracy_score,log_loss,recall_score
from sklearn.cross_validation import cross_val_score
from sklearn import tree
from IPython.display import Image
from sklearn.externals.six import StringIO
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
Street=[]
days=[]
District=[]


Time1=[]
Hour1=[]
Minute1=[]
Month1=[]
Year1=[]
Date1=[]
Seasons1=[]
WeekDays1=[]
TimeSlots1=[]
Category1=[]
Features1=[]
Latitude1=[]
Longitude1=[]
Street1=[]
days1=[]
District1=[]


def parse_time(x):
    DD=datetime.strptime(x,"%Y-%m-%d %H:%M:%S")
    hours=DD.hour#*60+DD.minute
    minutes=DD.minute
    date=DD.day
    month=DD.month
    year=DD.year
    if date in range(1,11):
        time=1
    elif date in range(11,21):
        time=2
    else:
        time=3
    return hours,minutes,date,month,year,time

df_train=pd.DataFrame.from_csv("train.csv",index_col=False,parse_dates=True)
df_test=pd.DataFrame.from_csv("test.csv",index_col=False,parse_dates=True)

def parse_weekday(x):
    Weekdays=["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
    for i,value in enumerate(Weekdays):
        if value==x:
            return i

def parse_district(x):
    district=['BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION','NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']
    for i,value in enumerate(district):
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
        return 0#"Winter"
    elif x in range(3,6):
        return 1#"Spring"
    elif x in range(6,9):
        return 2#"Summer"
    elif x in range(9,12):
        return 3#"Fall"

def Adaboost(TrainData,TestData):
    features=['Time','Season','Hour','Minute','District']

    #train,validation= train_test_split(TrainData, test_size=0.2)
    clf = AdaBoostClassifier(tree.DecisionTreeClassifier(),n_estimators=30)
    #clf=GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
    clf = clf.fit(TrainData[features], TrainData['Category'])
    predicted=np.array(clf.predict_proba(TestData[features]))
    model=clf.predict(TrainData[features])
    model1=clf.predict(TestData[features])

    #scores = cross_val_score(clf, validation[features], validation['Category'])
    #print "Scores mean is",scores.mean()
    #accuracy
    #print accuracy_score(train['Category'].values.tolist(),model)
    #print accuracy_score(validation['Category'].values.tolist(),model1)
    #print "Precision is ",precision_score(validation['Category'].values.tolist(),model1,average='macro')
    #print "Recall is ",recall_score(validation['Category'].values.tolist(),model1,average='macro')
    #print "Log loss is", log_loss(validation['Category'].values.tolist(),predicted,eps=1e-15, normalize=True, sample_weight=None)


    #writing to file
    Category_new=[]
    for i in range(0,len(model1)):
        Category_new.append(le_crime.classes_[model1[i]])
    result=pd.DataFrame(predicted, columns=le_crime.classes_)
    result['Predicted']=Category_new
    #result.to_csv('GradientBoost.csv', index = True, index_label = 'Id' )
    result.to_csv('AdaBoost.csv', index = True, index_label = 'Id' )

for allrows in df_train.values:
    myhour,myminute,mydate,mymonth,myyear,mytime = parse_time(allrows[0])

    Hour.append(myhour)
    Minute.append(myminute+100)
    Date.append(mydate)
    Month.append(mymonth)
    Time.append(mytime)
    Year.append(myyear)
    Seasons.append(parse_seasons(mymonth))
    TimeSlots.append(timeslots(myhour))
    days.append(parse_weekday(allrows[3]))
    District.append(parse_district(allrows[4]))
    if '/' in allrows[6]:
        Street.append(1)
    else:
        Street.append(0)
    if allrows[3] in ['Monday','Tuesday','Wednesaday','Thursday','Friday']:
        WeekDays.append(1)
    else:
        WeekDays.append(0)
    if int(allrows[7])==-120:
        Latitude.append(948)
    else:
        Latitude.append(int(((allrows[7]+122)*100)+1000))
    if int(allrows[8])==90:
        Longitude.append(2082)
    else:
        Longitude.append(int(((allrows[8]-37)*100)+2000))


le_crime = preprocessing.LabelEncoder()
crime = le_crime.fit_transform(df_train['Category'])


#adding to train_data
train_data=pd.concat([df_train['Address'],df_train['Category']],axis=1)
train_data['Category']=crime
train_data['Month']=Month
train_data['Date']=Date
train_data['Year']=Year
train_data['Hour']=Hour
train_data['Minute']=Minute
train_data['Season']=Seasons
train_data['Time']=Time
train_data['Street']=Street
train_data['Weekdays']=WeekDays
train_data['DayOfWeek']=days
train_data['District']=District
train_data['X']=Longitude
train_data['Y']=Latitude
DecisionTreeClassifier(train_data)


#for test data
for allrows in df_test.values:
    myhour,myminute,mydate,mymonth,myyear,mytime = parse_time(allrows[1])

    Hour1.append(myhour)
    Minute1.append(myminute+100)
    Date1.append(mydate)
    Month1.append(mymonth)
    Time1.append(mytime)
    Year1.append(myyear)
    weekday=parse_weekday(allrows[2])
    WeekDays1.append(weekday)
    Seasons1.append(parse_seasons(mymonth))
    TimeSlots1.append(timeslots(myhour))
    District1.append(parse_district(allrows[3]))
    if int(allrows[5])==-120:
        Latitude1.append(948)
    else:
        Latitude1.append(int(((allrows[5]+122)*100)+1000))
    if int(allrows[6])==90:
        Longitude1.append(2082)
    else:
        Longitude1.append(int(((allrows[6]-37)*100)+2000))

test_data=pd.concat([df_test['Address']],axis=1)
test_data['Month']=Month1
test_data['Date']=Date1
test_data['Year']=Year1
test_data['Hour']=Hour1
test_data['Minute']=Minute1
test_data['Season']=Seasons1
test_data['Time']=Time1
test_data['Weekdays']=WeekDays1
test_data['DayOfWeek']=WeekDays1
test_data['District']=District1
test_data['X']=Longitude1
test_data['Y']=Latitude1
Adaboost(train_data,test_data)
