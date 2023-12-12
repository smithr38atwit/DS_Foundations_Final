# Data Science Foundations - Final Project

## Introduction
The goal of this project was to determine if lap times of cars at a specific race track can be predicted based on basic information about the cars. Motor racing is a highly data driven sport, especially at the highest levels,
so the ability to collect and accurately analyze data to make improvements or predictions is crucial to victory. This project looks at multiple regression techniques, a simple neural net, and hyperparameter tuning with grid
search to produce the best result possible with limited data.

## Selection of Data
The data for this project is a small dataset of the fastest lap times for different cars at the Leguna Seca racetrack in California. The original data was scraped from [FastestLaps](https://fastestlaps.com/), and the csv used
in this project was downloaded from [Kaggle](https://www.kaggle.com/datasets/reggiejanmarcnillo/laguna-seca-lap-times-via-fastestlaps/).

The dataset has 187 samples and 13 features:
- Name
- CarLink
- Time
- PS
- Weight_KG
- Car_type
- Year
- Country_of_Origin
- Engine_type
- Displacement
- Transmission
- Engine_Layout
- Drivetrain

The goal was to predict "Time" (the lap time in seconds) using the other 12 features.

![image](https://github.com/smithr38atwit/DS_Foundations_Final/assets/54961768/9b3b0ee9-ecb9-4aaa-8d75-8d7089e12f07)

As seen in the sample above, much of this dataset is text. I was able to immediately identify a few columns that were irrelevant and could be dropped: Name, CarLink, Year, Country_of_Origin, and Car_type.
Name and CarLink (link to the cars page on FastestLaps) are clearly irrelevant, while the other 3 have no direct impact on car performance and thus could throw off the models by creating false correlations.


The rest of the text columns seemed relevant, so to determine if they were useful and able to be converted into numeric factors, I took a sum of the number of unique values in each column.

![image](https://github.com/smithr38atwit/DS_Foundations_Final/assets/54961768/803a0b9b-ff88-46f8-b74a-ecff5e2cd1bf)

Considering that there's only 187 samples, Engine_Type and Transmission have way too many unique values to be encoded and should be dropped. That leaves Time as the label, and PS (essentially equivalent to horsepower), Weight_KG, 
Displacement, Engine_Layout, and Drivetrain as the features.

With the columns selected, the last step to preparing the data was creating a column transformer that would preprocess the data in a pipeline before training each model. The first 3 features are continuous numerical values so the 
standard scalar was used. The last two are unordered categorical text data, so a one-hot encoder was used. This converted them each from a single column with 3 unique values, to 3 columns with binary values for a total of 9 feature columns.

![image](https://github.com/smithr38atwit/DS_Foundations_Final/assets/54961768/862bdc4b-5576-4702-abb7-9140c8806d22)

## Methods

Tools:
- Pandas and Scikit-Learn for data manipulation and regression models
- Keras for neural network
- Matplotlib for visualization

Regression Models:
- SVR
- Lasso
- ElasticNet

Neural Net: Sequential model with one hidden layer

Hyperparameter tuning: GridSearchCV

Metrics:
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)

## Results

### Regression Models

The first model tested was a SVR with a linear kernel, which had the following MAE and MSE:

![image](https://github.com/smithr38atwit/DS_Foundations_Final/assets/54961768/51a626b2-6d6b-4142-9756-6ae3e28763f7)

Both metrics were used because MSE is sensitive to outliers and gives a number that is hard to interpret without other results to compare to. MAE may not convey differences between models as well, but it is easy to understand since it's in the 
same units as the label (in this case seconds). The MAE of means that 3.86s was the average error; this is a long time in racing, but for this data it's not a terrible result. 

To visualize the error, I plotted the predicted values against the actual test values using lap time on the y-axis, and both PS and weight on the x-axis. These two features are logically the most 
impactful on a car's track time, so they were chosen as the features to compare.

![svr_linear_ps](https://github.com/smithr38atwit/DS_Foundations_Final/assets/54961768/e42ef324-4bb7-499d-a102-e7a03d900277)
![svr_linear_weight](https://github.com/smithr38atwit/DS_Foundations_Final/assets/54961768/ddfc896a-209a-4988-8f42-b07e6f235f52)

Next I tested lasso, which had an overall worse result:

![image](https://github.com/smithr38atwit/DS_Foundations_Final/assets/54961768/db77de19-e339-42c5-8e3d-2c3ae06a34bf)

The weight plot does not show much difference for this model, so it is not shown. The predictions in the PS plot, on the other hand, are much more tightly packed and clearly follow the PS in a linear fassion, indicating that the model may 
have relied too much on just the PS.

![lasso_ps](https://github.com/smithr38atwit/DS_Foundations_Final/assets/54961768/10a6b60d-e1af-4ccb-b521-f08a5085b695)

The last regression model tested was ElasticNetCV. This models performance was comparable to SVR and had almost identical looking plots.

![image](https://github.com/smithr38atwit/DS_Foundations_Final/assets/54961768/52111080-305b-430c-926b-5f5b022c6c71)

Lasso is meant for very few features and was performing worse than the other 2 models, so I assumed that it would not do much better when tuned and continued on to hyperparameter tuning with SVR and ElasticNetCV. However before tuning these models, 
I wanted to try using a simple nerual net to see if it would be difficult to get the same or better results than the base regression models. 

### Neural Net

I tested 2 different architectures, each with an input layer for the 9 feaures, a hidden layer with "relu" activation, 
and an output layer with a single neuron for the lap time prediction. I used 3 neurons in the hidden layer for the first model, and 15 in the second model.

![image](https://github.com/smithr38atwit/DS_Foundations_Final/assets/54961768/6514a858-b839-4034-aaaa-4aee078f8db8)
![image](https://github.com/smithr38atwit/DS_Foundations_Final/assets/54961768/2553d9c3-75d3-44e2-88d4-9383b3712a66)

The errors were similar for both, and were much worse than any of the regression models. To further compare the models and ensure they had run for enough epochs to converge, I plotted the epochs against the loss (MSE) of each model during training:

![nn_mse](https://github.com/smithr38atwit/DS_Foundations_Final/assets/54961768/23056a1b-0845-4077-b412-6903ac8b407a)

This shows that the curves for both had smoothed out and were unlikely to improve any more. The only difference between the smaller and larger number of neurons was that the second model converged faster, but it did not achieve a significantly different result.
The problem being tackled in this project is fairly simple, so a neural net is an overcomplicated and unnnecessary solution. This seems to be reflected by the inital errors, so it is not worth tuning to try to improve the performance.

### Hyperparameter Tuning

I used gridsearch to tune both SVR and ElasticNetCV in an effort to improve the base results. For tuning SVR, I used a small parameter grid with different kernel types and C values (regularization parameter):

![image](https://github.com/smithr38atwit/DS_Foundations_Final/assets/54961768/f7bace27-3185-4098-ba61-119cf6d2e6d5)

![image](https://github.com/smithr38atwit/DS_Foundations_Final/assets/54961768/76fc8031-9481-4fd0-bd06-e93241966593)

The best hyperparameters for SVR were an "rbf" kernel and c value of 10, resulting in a sizeable decrease in error.

![svr_grid_weight](https://github.com/smithr38atwit/DS_Foundations_Final/assets/54961768/cf8acbbc-e4a3-4dd9-b3f6-2bd49baf89c1)
![svr_grid_ps](https://github.com/smithr38atwit/DS_Foundations_Final/assets/54961768/e5b8c95b-f485-4691-9563-fdcffd59a981)

Looking at the error plots, the predictions seem to be slightly more spread out. Most noticeably a few points were pulled further away from the main cluster towards the outliers, reducing the effect these outliers were having on the MSE. This is the best model achieved, 
and although the MAE being a few seconds is still large, it is a much more viable model than before which proves the hypothesis that lap times can be accurately predicted with this data.

Tuning ElasticNetCV did not have the same affect and only showed very slight improvement in error:

![image](https://github.com/smithr38atwit/DS_Foundations_Final/assets/54961768/80f11551-002b-4cb9-bafc-1069830df4da)
