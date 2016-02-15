# San-Fransisco-Crime-Classification

Problem Description

From 1934 to 1963, San Francisco was infamous for housing some of the world's most notorious criminals on the inescapable island of Alcatraz.With rising wealth inequality, housing shortages, and a proliferation of expensive digital toys riding BART to work, there is no scarcity of crime in the city. So the problem here is, given the crime data report for 12 years, we should be able to predict the category of crime based on time and location.

2. Resources and Tools Used

2.1 Dataset

This dataset is taken from Kaggle which contains incidents derived from SFPD Crime Incident Reporting system.The dataset was provided by SF OpenData.
● The data ranges from : 1/1/2003­5/13/2015

● Train dataset size : 878050
● Test dataset size : 884263
● Crime Categories : 39
● Number of features : 9

The data fields are described below:
● Dates ​­ timestamp of the crime incident
● Category ­ category of the crime incident (only in train.csv). This is the target variable you are
going to predict.
● Descript​­ detailed description of the crime incident (only in train.csv)
● DayOfWeek​­ the day of the week
● PdDistrict ​­ name of the Police Department District
● Resolution​­ how the crime incident was resolved (only in train.csv)
● Address​­ the approximate street address of the crime incident
● X​­ Longitude
● Y ​­ Latitude

2.2 Tools
Scikit­learn​: It is a machine learning python library. It features various classification , clustering and
regression algorithms. It is designed to interoperate with the python numerical and scientific libraries
NumPy and SciPy.

Pandas: Software library written in python for data manipulation and analysis. Data structure called data
frame is used in this project for loading and generating csv files.It helps to read and write to csv files.

Matplotlib​: It is a plotting library for the Python programming language and its numerical mathematics
extension NumPy. It helps in plotting the graphs.

Seaborn: Seaborn is a Python visualization library based on matplotlib. It provides a high­level interface
for drawing attractive statistical graphics.
