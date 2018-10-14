<p align="center"><img height="200" src="http://www.esplore.polimi.it/wp-content/uploads/2017/05/polimi-logo-1-1.png"></p>


# RecSys Toolbox

## Overview

This is a set of recommender systems methods for the Kaggle competition of the Recommender Systems course in Politecnico di Milano, in which I take part as a single-member team called 'tms'.

The application domain of the [Recommender System 2017 Challenge Polimi](https://www.kaggle.com/c/recommender-system-2017-challenge-polimi) is a music streaming service, where users listen to tracks and create playlists of favorite ones. The main goal of the competition is to discover which track a user will likely add to a playlist, predicting a list of 5 tracks for a set of playlists.

## Data

In the Data folder, you can find the dataset provided for the competition:

* **train_final.csv** - the training set of interactions
* **tracks_final.csv** - supplementary information about the items
* **playlists_final.csv** - supplementary information about the users
* **target_playlists.csv** - the set of target playlists that will receive recommendations
* **target_tracks.csv** - the set of target items (tracks) to be recommended

## Evaluation

The evaluation metric for the competition is MAP@5 (Mean Average Precision) that takes into account the ranking of predictions. For instance, on a prediction based on 5 tracks, if only one of them is correct, MAP will give an higher value if the correct track is the first recommended rather than the second, the third and so on.

The average precision at 5, for a user is defined as:

<p align="center"><img height="75" src="https://github.com/tmscarla/RecSys/blob/master/img/ap.png"></p>

where P(k) is the precision at cut-off k, rel(i) is 1 if item in position k is relevant, 0 otherwise, and m is the number of relevant items in the test set. P(k) equals 0 if k-h item is not relevant.

The mean average precision for N users at position 5 is the average of the average precision of each user, i.e.,

<p align="center"><img height="75" src="https://github.com/tmscarla/RecSys/blob/master/img/map.png"></p>

# Get started

If you want to test any of the implemented recommenders using the parameters that I tuned to achive the best results, just open up Run.py and uncomment any of this line to launch the prediction of the recommender on the dataset.

Each of these functions takes one parameter, is_test, which is useful to determine if the prediction is used to be tested locally on the test set built with Evaluator.py, or to make a real prediction in the appropriate format.

```python
rs.hybrid_rec(is_test=True)
# rs.top_pop_rec()
# rs.item_based(is_test=True)
# rs.round_robin_rec(is_test=True, avg_mode=False)
# rs.round_robin_rec(is_test=True, avg_mode=True)
# rs.item_based(is_test=True)
# rs.SVD(is_test=True)
# rs.item_user_avg(is_test=True)
# rs.collaborative_filtering(is_test=True)
```

On the other hand, if you want to ajust any of the hyperparameters, you can open Recsys.py and change the parameters in the rec.fit() function used to train the recommender.

```python
from Builder import Builder
from Evaluator import Evaluator
from Recommenders import ItemBasedRec, CollaborativeFilteringRec, ItemUserAvgRec,\
    SlimBPRRec, SVDRec, RoundRobinRec, HybridRec, TopPopRec
    
def item_based(is_test):
    print('*** Item Based Recommender ***')

    b = Builder()
    ev = Evaluator()
    ev.split()
    rec = ItemBasedRec.ItemBasedRec()

    S_ICM = b.build_S_ICM_knn(b.build_ICM(), 250)

    rec.fit(ev.get_URM_train(), ev.get_target_playlists(), ev.get_target_tracks(), ev.num_playlists_to_test,
            S_ICM, is_test)
    train_df = rec.recommend()

    if is_test:
        map5 = ev.map5(train_df)
        print('ItemBased MAP@5:', map5)
    else:
        print('Prediction saved!')
        train_df.to_csv('ItemBased.csv', sep=',', index=False)
```

Furthermore, in the Builder.py are gathered all the prepreprocessing methods useful to create the training set and matrices, such as the User Rating Matrix (URM) or the Item Content Matrix (ICM).

# Results

I trained and test each recommender separately, finding out the the best results came from the Content Based recommender, the Collaborative Filtering recommender and the Slim BPR recommender.

According to their MAP@5, I combine them in two stages:
* Merge the models of Item Based and Collaborative Filtering recommenders on the 'avg' parameter
* Combine recommendation with the Slim BPR ones

This approach led to a **MAP@5 = 0.10205**, which is the best result that I have obtained.

The process can be summarized as follows:

<p align="center">
<img height="350" src="https://github.com/tmscarla/RecSys/blob/master/img/hybrid.png">
</p>

For the course I had also to present my work to Professor Cremonesi and to my colleagues. All the slides are in the file 'Slides.pdf', entirely in English.

## Requirements
| Package              | Version        |
| ---------------------|:--------------:|  
| **scikit-learn**     |   >= 0.19.1    |   
| **numpy**            |   >= 1.14      |   
| **scipy**            |   >= 1.0.0     |   
| **pandas**           |   >= 0.22.0    |  
| **fastFM**           |   >= 0.2.10    | 
| **tqdm**             |   >= 4.19.5    |  
