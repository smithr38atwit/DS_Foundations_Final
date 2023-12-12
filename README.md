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

