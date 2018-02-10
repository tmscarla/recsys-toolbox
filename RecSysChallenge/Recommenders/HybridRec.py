import numpy as np
import scipy as sc
import pandas as pd
from scipy import sparse
from pandas import DataFrame
from tqdm import tqdm
from sklearn import feature_extraction
from time import sleep
from scipy.sparse.linalg import svds
from Builder import Builder


"""
This hybrid recommender combines three different recommenders in these way:

    - Compute the weighted sum on avg of S_ICM and S_UCM similarity matrices
    - Calculate prediction of the new similarity matrix
    - Compute the weighted sum on alfa of the above prediction and the SlimBPR prediction

If is_test is true, return a dataframe ready to be evaluated with the Evaluator class,
otherwise return a dataframe in the submission format.
"""


class HybridRec(object):

    def fit(self, URM, target_playlists, target_tracks, num_playlist_to_recommend,
            S_ICM, S_UCM, Slim, is_test, alfa, avg):

        self.URM = URM
        self.target_playlists = target_playlists
        self.target_tracks = target_tracks
        self.num_playlist_to_recommend = num_playlist_to_recommend
        self.S_ICM = S_ICM
        self.S_UCM = S_UCM
        self.Slim = Slim
        self.is_test = is_test
        self.alfa = alfa
        self.avg = avg

    def recommend(self):
        # Compute the indices of the non-target playlists
        b = Builder()
        nontarget_indices = b.get_nontarget_indices(self.target_tracks)

        # Initialize the dataframe
        dataframe_list = []

        # Apply tfidf on the traspose of URM
        URM_T = self.URM.T
        URM_tfidf_T = feature_extraction.text.TfidfTransformer().fit_transform(URM_T)
        URM_tfidf = URM_tfidf_T.T
        URM_tfidf_csr = URM_tfidf.tocsr()

        # Weighted average of S_ICM and S_UCM
        S_avg = (self.avg * self.S_ICM) + ((1 - self.avg) * self.S_UCM)

        print('Predicting...', flush=True)
        for i in tqdm(range(0, self.num_playlist_to_recommend)):
            # Iterate over indices of target playlists
            index = b.get_target_playlist_index(self.target_playlists[i])

            # Compute the indices of the known tracks
            known_indices = np.nonzero(self.URM[index].toarray().flatten())

            # Calculate recommenders contributions
            avg_prediction = URM_tfidf_csr[index, :] * S_avg

            slimBPR_prediction = URM_tfidf_csr[index, :] * self.Slim

            # Weighted sum of recommendations
            URM_row = (self.alfa * avg_prediction) + ((1 - self.alfa) * slimBPR_prediction)

            # Make top-5 prediction
            URM_row_flatten = URM_row.toarray().flatten()
            top_5_indices = b.get_top_5_indices(URM_row_flatten, nontarget_indices, known_indices, [])
            top_5_tracks = b.get_top_5_tracks_from_indices(top_5_indices)
            top_5_tracks_string = ' '.join([str(i) for i in top_5_tracks])

            # Create dataset
            if self.is_test:
                dataframe_list.append([self.target_playlists[i], top_5_tracks])
            else:
                dataframe_list.append([self.target_playlists[i], top_5_tracks_string])

        dataframe = pd.DataFrame(dataframe_list, columns=['playlist_id', 'track_ids'])

        return dataframe
