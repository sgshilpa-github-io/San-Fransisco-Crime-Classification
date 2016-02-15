from datetime import datetime
import pandas as pd
from numpy import array
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
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

def DecisionTreeClassifier(TrainData,TestData):
    features=['DayOfWeek','Hour','Minute','District','Street']
    season=['Fall','Spring','Summer','Winter']
    district=['BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION','NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']
    days=['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday','Wednesday']
    Street=['Yes','No']
    week=['Weekday','Weekend']
    time=['first','second','third']
    features2 = [x for x in range(0,24)]
    Minute=[x for x in range(100,160)]
    latitude=[x for x in range(948,964)]
    longitude=[x for x in range(2070,2083)]
    #features=days+district+week+latitude+longitude


    #train,validation= train_test_split(TrainData, test_size=0.1)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(TrainData[features], TrainData['Category'])
    predicted=np.array(clf.predict_proba(TestData[features]))
    model=clf.predict(TrainData[features])
    model1=clf.predict(TestData[features])

    #scores = cross_val_score(clf, validation[features], validation['Category'])
    #print "Scores mean is",scores.mean()
    #accuracy
    #print accuracy_score(train['Category'].values.tolist(),model)
    #print accuracy_score(validation['Category'].values.tolist(),model1)
    #print "Log loss is", log_loss(validation['Category'].values.tolist(),predicted,eps=1e-15, normalize=True, sample_weight=None)
    #print "Precision is ",precision_score(validation['Category'].values.tolist(),model1,average='macro')
    #print "Recall is ",recall_score(validation['Category'].values.tolist(),model1,average='macro')

    Category_new=[]
    for i in range(0,len(model1)):
        Category_new.append(le_crime.classes_[model1[i]])

    #store result into file
    result=pd.DataFrame(predicted, columns=le_crime.classes_)
    result['Predicted']=Category_new
    result.to_csv('decision_test.csv', index = True, index_label = 'Id' )

    #visualization
    """dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    print "done"
    #Image(graph.create_png())
    print "done done"""

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
#district = pd.get_dummies(df_train['PdDistrict'])
#days = pd.get_dummies(df_train['DayOfWeek'])
#Seasons=pd.get_dummies(Seasons)
#Hour=pd.get_dummies(Hour)
#Minute=pd.get_dummies(Minute)
#Latitude=pd.get_dummies(Latitude)
#Longitude=pd.get_dummies(Longitude)
#Time=pd.get_dummies(Time)
#Street=pd.get_dummies(Street)
#WeekDays=pd.get_dummies(WeekDays)
#adding to train_data

train_data=pd.concat([df_train['Address'],df_train['X'],df_train['Y'],df_train['Category']],axis=1)
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


for allrows in df_test.values:
    myhour,myminute,mydate,mymonth,myyear,mytime = parse_time(allrows[1])

    Hour1.append(myhour)
    Minute1.append(myminute+100)
    Date1.append(mydate)
    Month1.append(mymonth)
    Time1.append(mytime)
    Year1.append(myyear)
    Seasons1.append(parse_seasons(mymonth))
    TimeSlots1.append(timeslots(myhour))
    days1.append(parse_weekday(allrows[2]))
    District1.append(parse_district(allrows[3]))
    if '/' in allrows[4]:
        Street1.append(1)
    else:
        Street1.append(0)
    if allrows[2] in ['Monday','Tuesday','Wednesaday','Thursday','Friday']:
        WeekDays1.append(1)
    else:
        WeekDays1.append(0)
    if int(allrows[5])==-120:
        Latitude1.append(948)
    else:
        Latitude1.append(int(((allrows[5]+122)*100)+1000))
    if int(allrows[6])==90:
        Longitude1.append(2082)
    else:
        Longitude1.append(int(((allrows[6]-37)*100)+2000))


le_crime = preprocessing.LabelEncoder()
crime = le_crime.fit_transform(df_train['Category'])
#district = pd.get_dummies(df_train['PdDistrict'])
#days = pd.get_dummies(df_train['DayOfWeek'])
#Seasons=pd.get_dummies(Seasons)
#Hour=pd.get_dummies(Hour)
#Minute=pd.get_dummies(Minute)
#Latitude=pd.get_dummies(Latitude)
#Longitude=pd.get_dummies(Longitude)
#Time=pd.get_dummies(Time)*
#Street=pd.get_dummies(Street)
#WeekDays=pd.get_dummies(WeekDays)
#adding to train_data

test_data=pd.concat([df_test['Address'],df_test['X'],df_test['Y']],axis=1)
test_data['Date']=Date1
test_data['Year']=Year1
test_data['Hour']=Hour1
test_data['Minute']=Minute1
test_data['Season']=Seasons1
test_data['Time']=Time1
test_data['Street']=Street1
test_data['Weekdays']=WeekDays1
test_data['DayOfWeek']=days1
test_data['District']=District1
DecisionTreeClassifier(train_data,test_data)
