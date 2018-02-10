import numpy as np
import pandas as pd
from tqdm import tqdm
from Builder import Builder
from SlimBPR import SlimBPR

"""
Recommender using the BPR - Bayesian Personalized Ranking with SGD. 
It builds up the similarity matrix using the method in SlimBPR.py.

If is_test is true, return a dataframe ready to be evaluated with the Evaluator class,
otherwise return a dataframe in the submission format.
"""

class SlimBPRRec(object):

    def fit(self, URM, target_playlists, target_tracks, num_playlist_to_recommend,
            learning_rate, epochs, positive_item_regularization,
            negative_item_regularization, knn, nnz, is_test):
        self.URM = URM
        self.target_playlists = target_playlists
        self.target_tracks = target_tracks
        self.num_playlist_to_recommend = num_playlist_to_recommend
        self.is_test = is_test

        # Compute similarity matrix
        self.Slim = SlimBPR(URM,
                            learning_rate,
                            epochs,
                            positive_item_regularization,
                            negative_item_regularization,
                            nnz).get_S_SLIM_BPR(knn)

    def recommend(self):
        b = Builder()

        nontarget_indices = b.get_nontarget_indices(self.target_tracks)

        dataframe_list = []

        print('Predicting...', flush=True)
        for i in tqdm(range(0, self.num_playlist_to_recommend)):
            # Iterate over indices of target playlists
            index = b.get_target_playlist_index(self.target_playlists[i])

            # Compute relevant indices for the prediction
            known_indices = np.nonzero(self.URM[index].toarray().flatten())

            URM_row = self.URM[index, :] * self.Slim

            # Make prediction
            URM_row_flatten = URM_row.toarray().flatten()
            top_5_indices = b.get_top_5_indices(URM_row_flatten, nontarget_indices, known_indices, owner_indices)
            top_5_tracks = b.get_top_5_tracks_from_indices(top_5_indices)
            top_5_tracks_string = ' '.join([str(i) for i in top_5_tracks])

            # Create dataset
            if self.is_test:
                dataframe_list.append([self.target_playlists[i], top_5_tracks])
            else:
                dataframe_list.append([self.target_playlists[i], top_5_tracks_string])

        dataframe = pd.DataFrame(dataframe_list, columns=['playlist_id', 'track_ids'])

        return dataframe
