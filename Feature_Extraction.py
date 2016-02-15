__author__ = 'shilpagulati'

from datetime import datetime
import pandas as pd
from numpy import array
from sklearn.naive_bayes import BernoulliNB,GaussianNB
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import timeit
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score,accuracy_score,log_loss,recall_score

#train data
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

#test data storage
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

category=['KIDNAPPING', 'WEAPON LAWS', 'SECONDARY CODES', 'WARRANTS', 'PROSTITUTION', 'EMBEZZLEMENT', 'LOITERING', 'SUICIDE', 'DRIVING UNDER THE INFLUENCE', 'SEX OFFENSES FORCIBLE', 'ROBBERY', 'BURGLARY', 'SUSPICIOUS OCC', 'FAMILY OFFENSES', 'BRIBERY', 'FORGERY/COUNTERFEITING', 'BAD CHECKS', 'DRUNKENNESS', 'GAMBLING', 'OTHER OFFENSES', 'RECOVERED VEHICLE', 'FRAUD', 'ARSON', 'DRUG/NARCOTIC', 'TRESPASS', 'LARCENY/THEFT', 'VANDALISM', 'NON-CRIMINAL', 'EXTORTION', 'PORNOGRAPHY/OBSCENE MAT', 'LIQUOR LAWS', 'SEX OFFENSES NON FORCIBLE', 'TREA', 'VEHICLE THEFT', 'STOLEN PROPERTY', 'ASSAULT', 'MISSING PERSON', 'DISORDERLY CONDUCT', 'RUNAWAY']
category1=['KIDNAPPING', 'WEAPON LAWS', 'SECONDARY CODES', 'WARRANTS', 'PROSTITUTION', 'EMBEZZLEMENT', 'LOITERING', 'SUICIDE', 'DRIVING UNDER THE INFLUENCE', 'SEX OFFENSES FORCIBLE', 'ROBBERY', 'BURGLARY', 'SUSPICIOUS OCC', 'FAMILY OFFENSES', 'BRIBERY', 'FORGERY/COUNTERFEITING', 'BAD CHECKS', 'DRUNKENNESS', 'GAMBLING', 'OTHER OFFENSES', 'RECOVERED VEHICLE', 'FRAUD', 'ARSON', 'DRUG/NARCOTIC', 'TRESPASS', 'LARCENY/THEFT', 'VANDALISM', 'NON-CRIMINAL', 'EXTORTION', 'PORNOGRAPHY/OBSCENE MAT', 'LIQUOR LAWS', 'SEX OFFENSES NON FORCIBLE', 'TREA', 'VEHICLE THEFT', 'STOLEN PROPERTY', 'ASSAULT', 'MISSING PERSON', 'DISORDERLY CONDUCT', 'RUNAWAY']



#read test and train files
df_train=pd.DataFrame.from_csv("train.csv",index_col=False,parse_dates=True)
df_test=pd.DataFrame.from_csv("test.csv",index_col=False,parse_dates=True)

start = timeit.default_timer()
#parse time into month, date, hour, minut, second
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


def parse_month(x):
    month_names=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    for i,value in enumerate(month_names):
        if (i+1)==x:
            return value

#naive bayes
def naive_bayes(train,validation):

    #features
    season=['Fall','Spring','Summer','Winter']
    #season=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    district=['BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION','NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']
    time=['first','second','third']
    features2 = [x for x in range(0,24)]
    Minute=[x for x in range(100,160)]

    features=district+time+Minute+season+features2

    #split set into train, validation
    train,validation= train_test_split(train, train_size=0.9)
    model = BernoulliNB()
    model.fit(train[features],train['Category'])

    #time calculation
    stop = timeit.default_timer()
    print "Runnin  time naive bayes is ", stop-start

    predicted = np.array(model.predict_proba(validation[features]))
    model1=model.predict(validation[features])
    model2=model.predict(train[features])

    print "-----------------------------Naive Bayes----------------------------------------------------------------------------"
    print "Precision is ",precision_score(validation['Category'].values.tolist(),model1,average='macro')
    print "Recall is ",recall_score(validation['Category'].values.tolist(),model1,average='macro')
    print "Accuracy is ", accuracy_score(validation['Category'].values.tolist(),model1)
    print "Training Accuracy is ", accuracy_score(train['Category'].values.tolist(),model2)
    Category_new=[]
    for i in range(0,len(model1)):
        Category_new.append(le_crime.classes_[model1[i]])

    #store result into file
    result=pd.DataFrame(predicted, columns=le_crime.classes_)
    result['Predicted']=Category_new
    result.to_csv('naiveBayes_test.csv', index = True, index_label = 'Id' )


    #log loss function
    print "Log loss is", log_loss(validation['Category'],predicted,eps=1e-15, normalize=True, sample_weight=None)


#for train data
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
    if allrows[3] in ['Monday','Tuesday','Wednesaday','Thursday','Friday']:
        WeekDays.append("Weekday")
    else:
        WeekDays.append("Weekend")
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
district = pd.get_dummies(df_train['PdDistrict'])
days = pd.get_dummies(df_train['DayOfWeek'])
Seasons=pd.get_dummies(Seasons)
Hour=pd.get_dummies(Hour)
Minute=pd.get_dummies(Minute)
Latitude=pd.get_dummies(Latitude)
Longitude=pd.get_dummies(Longitude)
Time=pd.get_dummies(Time)


#adding to train_data
train_data=pd.concat([district,days,Seasons,Hour,Minute,Longitude,Latitude,Time],axis=1)
train_data['Category']=crime


#for test data
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
    if int(allrows[5])==-120:
        Latitude1.append(948)
    else:
        Latitude1.append(int(((allrows[5]+122)*100)+1000))
    if int(allrows[6])==90:
        Longitude1.append(2082)
    else:
        Longitude1.append(int(((allrows[6]-37)*100)+2000))

district1 = pd.get_dummies(df_test['PdDistrict'])
days1 = pd.get_dummies(df_test['DayOfWeek'])
Seasons1=pd.get_dummies(Seasons1)
Hour1=pd.get_dummies(Hour1)
Minute1=pd.get_dummies(Minute1)
Latitude1=pd.get_dummies(Latitude1)
Longitude1=pd.get_dummies(Longitude1)
Time1=pd.get_dummies(Time1)
WeekDays1=pd.get_dummies(WeekDays1)
test_data=pd.concat([district1,days1,Seasons1,Hour1,Minute1,Longitude1,Latitude1,Time1],axis=1)
print Seasons
print Seasons1

naive_bayes(train_data,test_data)
