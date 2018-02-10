import numpy as np
import scipy as sc
import pandas as pd
from tqdm import tqdm
from Builder import Builder


"""
This recommender computes separately the prediction from the S_ICM and the S_UCM,
then it performs a weighted sum of the components according to the alfa parameter.

If is_test is true, return a dataframe ready to be evaluated with the Evaluator class,
otherwise return a dataframe in the submission format.
"""


class ItemUserAvgRec(object):

    def fit(self, URM, target_playlists, target_tracks, num_playlist_to_recommend,
            S_ICM, S_UCM, is_test, alfa):

        self.URM = URM
        self.target_playlists = target_playlists
        self.target_tracks = target_tracks
        self.num_playlist_to_recommend = num_playlist_to_recommend
        self.S_ICM = S_ICM
        self.S_UCM = S_UCM
        self.is_test = is_test
        self.alfa = alfa

    def recommend(self):
        # Compute the indices of the non-target playlists
        b = Builder()
        nontarget_indices = b.get_nontarget_indices(self.target_tracks)

        # Initialize the dataframe
        dataframe_list = []

        print('Predicting...', flush=True)
        for i in tqdm(range(0, self.num_playlist_to_recommend)):

            # Iterate over indices of target playlists
            index = b.get_target_playlist_index(self.target_playlists[i])

            # Compute the indices of the known tracks
            known_indices = np.nonzero(self.URM[index].toarray().flatten())

            # Make single predictions
            ucm_pred = self.URM[index, :] * self.S_UCM
            icm_pred = self.URM[index, :] * self.S_ICM

            # Weighted sum of predictions
            URM_row = (self.alfa * icm_pred) + ((1 - self.alfa) * ucm_pred)

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
