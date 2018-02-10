
# RecSys

## Overview

This is a set of recommender systems methods for the Kaggle competition of the Recommender Systems course in Politecnico di Milano.

The application domain of the [Recommender System 2017 Challenge Polimi](https://www.kaggle.com/c/recommender-system-2017-challenge-polimi) is a music streaming service, where users listen to tracks and create playlists of favorite ones. The main goal of the competition is to discover which track a user will likely add to a playlist, predicting a list of 5 tracks for a set of playlists.

## Evaluation

The evaluation metric for the competition is MAP@5 (Mean Average Precision)

The average precision at 5, for a user is defined as:

...

where P(k) is the precision at cut-off k, rel(i) is 1 if item in position k is relevant, 0 otherwise, and m is the number of relevant items in the test set. P(k) equals 0 if k-h item is not relevant.

The mean average precision for N users at position 5 is the average of the average precision of each user, i.e.,

## Requirements
| Package              | Version        |
| ---------------------|:--------------:|  
| **scikit-learn**     |   >= 0.19.1    |   
| **numpy**            |   >= 1.14      |   
| **scipy**            |   >= 1.0.0     |   
| **pandas**           |   >= 0.22.0    |  
| **fastFM**           |   >= 0.2.10    | 
| **tqdm**             |   >= 4.19.5    |  
