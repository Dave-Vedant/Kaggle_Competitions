# Digital Engagement Prediction of MLB dataset

## Data Source: 
- [Kaggle Competition Dataset](https://www.kaggle.com/c/mlb-player-digital-engagement-forecasting/data) by Google Cloud

## Dataset Explaination:
You are tasked with forecasting four different measures of engagement (target1-target4) for a subset of MLB players who are active in the 2021 season. The data contains a set of static files that do not change with time (players.csv, teams.csv, seasons.csv, awards.csv) as well as daily data (train.csv) which is grouped by day. When predicting on a given date, you are forecasting the target variables for the next day (i.e. for date d, you're predicting the engagement for day d+1).

### Dataset Main Files:
- train.csv - the training set. This dataset has nested json format. This includes detail information of 11 different sets.
- example_test.csv - an example in the form of the test set that you’ll be evaluated on.
- example_sample_submission.csv - A sample submission file in the correct format based on the example test set.
- awards.csv - A collection of awards given out prior to 01/01/2018 (the first date in train.csv).
- players.csv - Library containing high level information about all MLB players in this dataset.
- seasons.csv - Information about start and end dates of all seasons in this dataset.
- teams.csv - Library containing high level information about all MLB teams.

For further information please visit Kaggle [Dataset](https://www.kaggle.com/c/mlb-player-digital-engagement-forecasting/data)

### Project Notebook Summary:
1. Exploratory Data Analytics:

