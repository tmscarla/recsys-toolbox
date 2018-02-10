from Builder import Builder
from Evaluator import Evaluator
from Recommenders import ItemBasedRec, CollaborativeFilteringRec, ItemUserAvgRec,\
    SlimBPRRec, SVDRec, RoundRobinRec, HybridRec, TopPopRec
import SlimBPR

"""
This file is a control panel to test or make predictions with provided recommenders.
All the recommenders are gathered in the "Recommenders" folder.

Each one needs to be trained at first, with the fit() function and then can make its
prediction with the recommend() function.

If is_test is true, the dataset will be split into training set (80%) and test set (20%)
and the MAP@5 will be computed on the test set.
Otherwise, if is false, a .csv file with the prediction will be produced.
"""


def top_pop_rec():
    print('*** Top Popular Recommender ***')

    rec = TopPopRec.TopPopRec()

    train_df = rec.recommend()
    train_df.to_csv('TopPopular.csv', sep=',', index=False)


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


def collaborative_filtering(is_test):
    print('*** Test Collaborative Filtering Recommender ***')

    b = Builder()
    ev = Evaluator()
    ev.split()
    rec = CollaborativeFilteringRec.CollaborativeFilteringRec()

    S_UCM = b.get_S_UCM_KNN(b.get_UCM(b.get_URM()), 500)

    rec.fit(ev.get_URM_train(), ev.get_target_playlists(), ev.get_target_tracks(), ev.num_playlists_to_test,
            S_UCM, True)
    train_df = rec.recommend()

    if is_test:
        map5 = ev.map5(train_df)
        print('CollaborativeFiltering MAP@5:', map5)
    else:
        print('Prediction saved!')
        train_df.to_csv('CollaborativeFiltering.csv', sep=',', index=False)


def item_user_avg(is_test):
    print('*** Test Item User Avg Recommender ***')

    b = Builder()
    ev = Evaluator()
    ev.split()
    rec = ItemUserAvgRec.ItemUserAvgRec()

    S_ICM = b.build_S_ICM_knn(b.build_ICM(), 250)
    S_UCM = b.get_S_UCM_KNN(b.get_UCM(b.get_URM()), 500)

    rec.fit(ev.get_URM_train(), ev.get_target_playlists(), ev.get_target_tracks(), ev.num_playlists_to_test,
            S_ICM, S_UCM, True, 0.80)

    train_df = rec.recommend()

    if is_test:
        map5 = ev.map5(train_df)
        print('ItemUserAvg MAP@5:', map5)
    else:
        print('Prediction saved!')
        train_df.to_csv('ItemUserAvg.csv', sep=',', index=False)


def slim_BPR(is_test):
    print('*** Test Slim BPR Recommender ***')

    ev = Evaluator()
    ev.split()
    rec = SlimBPRRec.SlimBPRRec()

    rec.fit(ev.get_URM_train(), ev.get_target_playlists(), ev.get_target_tracks(), ev.num_playlists_to_test, 0.1, 1,
            1.0, 1.0, 1000, 1, is_test)
    train_df = rec.recommend()

    if is_test:
        map5 = ev.map5(train_df)
        print('SlimBPR MAP@5:', map5)
    else:
        print('Prediction saved!')
        train_df.to_csv('SlimBPR.csv', sep=',', index=False)


def SVD(is_test):
    print('*** Test SVD Recommender ***')

    b = Builder()
    ev = Evaluator()
    ev.split()
    rec = SVDRec.SVDRec()

    rec.fit(ev.get_URM_train(), ev.get_target_playlists(), ev.get_target_tracks(), ev.num_playlists_to_test,
            b.build_ICM(), k=100, knn=250, is_test=is_test)
    train_df = rec.recommend()

    if is_test:
        map5 = ev.map5(train_df)
        print('SlimBPR MAP@5:', map5)
    else:
        print('Prediction saved!')
        train_df.to_csv('SlimBPR.csv', sep=',', index=False)


def round_robin_rec(is_test, avg_mode):
    print('*** Test Round Robin Recommender ***')

    b = Builder()
    ev = Evaluator()
    ev.split()
    rec = RoundRobinRec.RoundRobinRec()

    S_ICM = b.build_S_ICM_knn(b.build_ICM(), 250)
    S_UCM = b.get_S_UCM_KNN(b.get_UCM(ev.get_URM_train()), 500)
    Slim =  SlimBPR.SlimBPR(ev.get_URM_train()).get_S_SLIM_BPR(500)

    rec.fit(ev.get_URM_train(), ev.get_target_playlists(), ev.get_target_tracks(), ev.num_playlists_to_test,
            S_ICM, S_UCM, Slim, is_test, mode="jump", a=3, b=1, c=1)

    if avg_mode:
        train_df = rec.recommend_avg()
    else:
        train_df = rec.recommend_rr()

    if is_test:
        map5 = ev.map5(train_df)
        print('RoundRobin MAP@5:', map5)
    else:
        print('Prediction saved!')
        train_df.to_csv('RoundRobin.csv', sep=',', index=False)


def hybrid_rec(is_test):
    print('*** Test Hybrid Recommender ***')

    b = Builder()
    ev = Evaluator()
    ev.split()
    rec = HybridRec.HybridRec()

    S_ICM = b.build_S_ICM_knn(b.build_ICM(), 250)
    S_UCM = b.get_S_UCM_KNN(b.get_UCM(ev.get_URM_train()), 500)
    Slim = SlimBPR.SlimBPR(ev.get_URM_train()).get_S_SLIM_BPR(500)

    rec.fit(ev.get_URM_train(), ev.get_target_playlists(), ev.get_target_tracks(), ev.num_playlists_to_test,
            S_ICM, S_UCM, Slim, True, 0.20, 0.74)

    train_df = rec.recommend()

    if is_test:
        map5 = ev.map5(train_df)
        print('Hybrid MAP@5:', map5)
    else:
        print('Prediction saved!')
        train_df.to_csv('Hybrid.csv', sep=',', index=False)
