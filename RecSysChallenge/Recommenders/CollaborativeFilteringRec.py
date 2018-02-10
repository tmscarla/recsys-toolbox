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
from MFBPR import MFBPR

"""
Collaborative filtering recommender with cosine similarity.
The similarity matrix S_UCM is computed with the dot product between the UCM with
TfIdf and its transpose. 

If is_test is true, return a dataframe ready to be evaluated with the Evaluator class,
otherwise return a dataframe in the submission format.
"""


class CollaborativeFilteringRec(object):
    def __init__(self):
        self.URM = None
        self.target_playlists = None
        self.target_tracks = None
        self.num_playlist_to_recommend = 0
        self.S_UCM = None
        self.is_test = False

    def fit(self, URM, target_playlists, target_tracks, num_playlist_to_recommend,
            S_UCM, is_test):

        self.URM = URM
        self.target_playlists = target_playlists,
        self.target_tracks = target_tracks,
        self.num_playlist_to_recommend = num_playlist_to_recommend
        self.S_UCM = S_UCM
        self.is_test = is_test

    def recommend(self):
        # Compute the indices of the non-target playlists
        b = Builder()
        nontarget_indices = b.get_nontarget_indices(self.target_tracks)

        # Initialize the dataframe
        dataframe_list = []


        print('Predicting...', flush=True)
        for i in tqdm(range(0, self.num_playlist_to_recommend)):

            # Iterate over indices of target playlists
            index = b.get_target_playlist_index(self.target_playlists[0][i])

            # Compute the indices of the known tracks
            known_indices = np.nonzero(self.URM[index].toarray().flatten())

            # Calculate a row of the new URM
            URM_row = self.URM[index, :] * self.S_UCM

            # Make prediction
            URM_row_flatten = URM_row.toarray().flatten()
            top_5_indices = b.get_top_5_indices(URM_row_flatten, nontarget_indices, known_indices, [])
            top_5_tracks = b.get_top_5_tracks_from_indices(top_5_indices)
            top_5_tracks_string = ' '.join([str(i) for i in top_5_tracks])

            # Create dataset
            if self.is_test:
                dataframe_list.append([self.target_playlists[0][i], top_5_tracks])
            else:
                dataframe_list.append([self.target_playlists[0][i], top_5_tracks_string])

        dataframe = pd.DataFrame(dataframe_list, columns=['playlist_id', 'track_ids'])

        return dataframe
