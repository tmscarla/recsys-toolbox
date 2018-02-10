import numpy as np
import pandas as pd
from tqdm import tqdm
from Builder import Builder

"""
Item based recommender with cosine similarity.
Suggests tracks according to the similarity of the other tracks in the playlist.

If is_test is true, return a dataframe ready to be evaluated with the Evaluator class,
otherwise return a dataframe in the submission format.
"""


class ItemBasedRec(object):

    def fit(self, URM, target_playlists, target_tracks, num_playlist_to_recommend,
            S_ICM, is_test):

        self.URM = URM
        self.target_playlists = target_playlists,
        self.target_tracks = target_tracks,
        self.num_playlist_to_recommend = num_playlist_to_recommend
        self.S_ICM = S_ICM
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
            URM_row = self.URM[index, :] * self.S_ICM

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

