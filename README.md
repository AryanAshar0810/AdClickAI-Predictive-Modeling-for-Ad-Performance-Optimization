## Abstract
In the current digital era, advertising has grown to be both commonplace and essential, and companies are always looking for ways to maximize the impact of their online advertising campaigns. This abstract presents a novel method for achieving this objective by applying machine learning techniques. Through the prediction of two crucial metrics—**Click-Through Rates (CTR)** and **Conversion Rates**—this initiative seeks to provide marketers with insightful data regarding the effectiveness of their ads. Sophisticated machine learning models can predict the chance that a person will click on an advertisement and then convert it by analyzing past data. By utilising algorithms like decision trees, classification models, this predictive framework guarantees that advertising campaigns are customized to target the most responsive demographic.


## Problem Statement
Businesses often struggle to determine the effectiveness of their advertisements, leading to inefficient marketing spending. This project aims to address this issue by developing a machine-learning model capable of predicting ad performance. By analyzing historical data and relevant features, the model will forecast metrics like click-through rates (CTR) or conversion rates. The goal is to provide businesses with actionable insights to optimize their ad campaigns and improve marketing ROI.
The effectiveness of particular advertisements has a significant influence on public relations overall. According to the research, simply raising someone’s awareness of certain goods, events, and brands increases the likelihood that they will buy those goods, attend those events, or support those brands. Additionally, a man’s odds of making a direct item commitment greatly increase if an advertisement manages to capture his attention to the extent that he or she has a prompt, good response to it.


## Goal 
Main goal is to determine the accuracy of performance of various machine learning models like **Naive Bayes, Decision Tree, K-Nearest Neighbors** on whether the user is likely to click on the Ad or not.


## Proposed Model
The ”Ad Performance Prediction with Machine Learning” project is designed to develop a machine learning model that predicts the effectiveness of online advertisements. By analysing historical ad data, user interactions, and various ad content attributes, this project aims to provide advertisers with insights into which ads are likely to perform well, thereby optimising advertising campaigns and maximising return on investment.

1. Data Collection and Preprocessing:
   * Loading Data: The dataset is loaded from a CSV file named "advertising.csv".
   
   * Checking for Duplicates: The dataset is checked for duplicate records and none are found.
   
   * Attribute Type Classification: The attributes are classified into numeric and categorical types.
   
   * Exploratory Data Analysis (EDA): Majority of internet users are aged between 26 to 42 years. Teenagers and individuals aged between 20-40 earn the highest income, 
    ranging between 50k-70k. Age group of 25-40 spends the most time on the internet. More females have clicked on ads compared to males. Initial statistics on the "Clicked 
    on Ad" feature are examined. Pairplot is used to visualize relationships between different features.

2. Cleaning
   * Missing Data: No missing data found.
     
   * Dropping Features: "Ad Topic Line" is dropped. "City" and "Country" are converted into numerical features using categorical encoding. "Timestamp" is converted into 
     numerical values, considering month and hour as features.

3. Modelling
   * Splitting Dataset: The dataset is split into training and testing sets with a ratio of 70:30.
     
   * Naive Bayes Model: A Naive Bayes model (GaussianNB) is implemented using GaussianNB from sklearn. The model is trained using the training data. Predictions are made on 
     the test data.
     
   * Decision Tree Model: A decision tree model is implemented using DecisionTreeClassifier from sklearn. The model is trained using the training data. Predictions are made 
    on the test data.

   * K-Nearest Neighbors Model: A KNN model is implemented using KNeighborsClassifier from sklearn. The model is trained using the training data. Predictions are made on the 
    test data.

4. Model Evaluation
   * Evaluate model performance using appropriate metrics, such as accuracy, precision, recall, F1 score, or confusion matrix, depending on the prediction task (classification or regression).

5. Testing & Results
   * Accuracy Calculation: Accuracy scores are calculated for each model using the accuracy score from sklearn. Naive Bayes: 96.0%,  Decision Tree: 93.33%, and KNN: 68%.


