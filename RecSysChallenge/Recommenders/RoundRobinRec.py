import numpy as np
import scipy as sc
import pandas as pd
from scipy import sparse
from pandas import DataFrame
from tqdm import tqdm
from sklearn import feature_extraction
from Builder import Builder

"""
This is a hybrid recommender which combines predictions of an item-based,
a collaborative filtering and a Slim BPR recommenders.

It works in two different ways:

- Round Robin
    * Standard: pick the best track from each recommendation
    * Jump: if the tracks is already selected, jump to the next recommendation
    * Mono: pick exactly one track for each recommendation

- Rankin Average: for each track, compute the average of rankings and pick the 
  top 5 tracks according to this value. 

If is_test is true, return a dataframe ready to be evaluated with the Evaluator class,
otherwise return a dataframe in the submission format.
"""


class RoundRobinRec(object):

    def fit(self, URM, target_playlists, target_tracks, num_playlist_to_recommend,
            S_ICM, S_UCM, Slim, is_test, mode, a, b, c):

        self.URM = URM
        self.target_playlists = target_playlists
        self.target_tracks = target_tracks
        self.num_playlist_to_recommend = num_playlist_to_recommend
        self.S_ICM = S_ICM
        self.S_UCM = S_UCM
        self.Slim = Slim
        self.is_test = is_test
        self.mode = mode
        self.a = a
        self.b = b
        self.c = c

    def recommend_rr(self):
        builder = Builder()
        nontarget_indices = builder.get_nontarget_indices(self.target_tracks)

        URM_T = self.URM.T
        URM_tfidf_T = feature_extraction.text.TfidfTransformer().fit_transform(URM_T)
        URM_tfidf = URM_tfidf_T.T
        URM_tfidf_csr = URM_tfidf.tocsr()

        dataframe_list = []

        print('Predicting round_robin with mode =', self.mode, '...', flush=True)
        for i in tqdm(range(0, self.num_playlist_to_recommend)):

            # Iterate over indices of target playlists
            index = builder.get_target_playlist_index(self.target_playlists[i])

            # Compute relevant indices for the prediction
            known_indices = np.nonzero(self.URM[index].toarray().flatten())

            # Calculate recommenders contributions
            icm_prediction = self.URM[index, :] * self.S_ICM
            icm_prediction_flatten = icm_prediction.toarray().flatten()
            icm_prediction_flatten[known_indices] = 0
            icm_prediction_flatten[nontarget_indices] = 0

            ucm_prediction = self.URM[index, :] * self.S_UCM
            ucm_prediction_flatten = ucm_prediction.toarray().flatten()
            ucm_prediction_flatten[known_indices] = 0
            ucm_prediction_flatten[nontarget_indices] = 0

            slimBPR_prediction = URM_tfidf_csr[index, :] * self.Slim
            slimBPR_prediction_flatten = slimBPR_prediction.toarray().flatten()
            slimBPR_prediction_flatten[known_indices] = 0
            slimBPR_prediction_flatten[nontarget_indices] = 0

            # Round Robin prediction
            top_5_indices = self.round_robin(icm_prediction_flatten,
                                             ucm_prediction_flatten,
                                             slimBPR_prediction_flatten,
                                             self.mode, self.a, self.b, self.c)
            top_5_tracks = builder.get_top_5_tracks_from_indices(top_5_indices)
            top_5_tracks_string = ' '.join([str(i) for i in top_5_tracks])

            # Create dataset
            if self.is_test:
                dataframe_list.append([self.target_playlists[i], top_5_tracks])
            else:
                dataframe_list.append([self.target_playlists[i], top_5_tracks_string])

        dataframe = pd.DataFrame(dataframe_list, columns=['playlist_id', 'track_ids'])

        return dataframe

    def recommend_avg(self):
        builder = Builder()

        nontarget_indices = builder.get_nontarget_indices(self.target_tracks)

        # Apply tfidf on the transpose of the URM
        URM_T = self.URM.T
        URM_tfidf_T = feature_extraction.text.TfidfTransformer().fit_transform(URM_T)
        URM_tfidf = URM_tfidf_T.T
        URM_tfidf_csr = URM_tfidf.tocsr()

        dataframe_list = []

        print('Predicting avg_prediction...', flush=True)
        for i in tqdm(range(0, self.num_playlist_to_recommend)):

            # Iterate over indices of target playlists
            index = builder.get_target_playlist_index(self.target_playlists[i])

            # Compute relevant indices for the prediction
            known_indices = np.nonzero(self.URM[index].toarray().flatten())

            # Calculate recommenders contributions
            icm_prediction = self.URM[index, :] * self.S_ICM
            icm_prediction_flatten = icm_prediction.toarray().flatten()
            icm_prediction_flatten[known_indices] = 0
            icm_prediction_flatten[nontarget_indices] = 0

            ucm_prediction = self.URM[index, :] * self.S_UCM
            ucm_prediction_flatten = ucm_prediction.toarray().flatten()
            ucm_prediction_flatten[known_indices] = 0
            ucm_prediction_flatten[nontarget_indices] = 0

            slimBPR_prediction = URM_tfidf_csr[index, :] * self.Slim
            slimBPR_prediction_flatten = slimBPR_prediction.toarray().flatten()
            slimBPR_prediction_flatten[known_indices] = 0
            slimBPR_prediction_flatten[nontarget_indices] = 0

            # Round Robin prediction
            top_5_indices = self.compute_avg_prediction(icm_prediction_flatten,
                                                        ucm_prediction_flatten,
                                                        slimBPR_prediction_flatten)
            top_5_tracks = builder.get_top_5_tracks_from_indices(top_5_indices)
            top_5_tracks_string = ' '.join([str(i) for i in top_5_tracks])

            # Create dataset
            if self.is_test:
                dataframe_list.append([self.target_playlists[i], top_5_tracks])
            else:
                dataframe_list.append([self.target_playlists[i], top_5_tracks_string])

        dataframe = pd.DataFrame(dataframe_list, columns=['playlist_id', 'track_ids'])

        return dataframe


    # SUPPORT FUNCTIONS

    def round_robin(self, icm_prediction_flatten, ucm_prediction_flatten, slimBPR_prediction_flatten,
                    mode, a, b, c):
        icm_argsort = icm_prediction_flatten.argsort()[::-1]
        ucm_argsort = ucm_prediction_flatten.argsort()[::-1]
        slim_argsort = slimBPR_prediction_flatten.argsort()[::-1]

        i = 0
        a_last_index = 0
        b_last_index = 0
        c_last_index = 0
        top_5_indices = []

        if mode == "std":
            while i < 5:
                a_i = 0
                b_i = 0
                c_i = 0

                while a_i != a:
                    if icm_argsort[a_last_index] not in top_5_indices:
                        top_5_indices.append(icm_argsort[a_last_index])
                        a_i += 1
                        i += 1
                    a_last_index += 1

                while b_i != b:
                    if ucm_argsort[b_last_index] not in top_5_indices:
                        top_5_indices.append(ucm_argsort[b_last_index])
                        b_i += 1
                        i += 1
                    b_last_index += 1

                while c_i != c:
                    if slim_argsort[c_last_index] not in top_5_indices:
                        top_5_indices.append(slim_argsort[c_last_index])
                        c_i += 1
                        i += 1
                    c_last_index += 1

        if mode == "jump":
            while i < 5:
                for a_i in range(0, a):
                    if icm_argsort[a_last_index] not in top_5_indices:
                        top_5_indices.append(icm_argsort[a_last_index])
                        i += 1
                    a_last_index += 1

                for b_i in range(0, b):
                    if ucm_argsort[b_last_index] not in top_5_indices:
                        top_5_indices.append(ucm_argsort[b_last_index])
                        i += 1
                    b_last_index += 1

                for c_i in range(0, c):
                    if slim_argsort[c_last_index] not in top_5_indices:
                        top_5_indices.append(slim_argsort[c_last_index])
                        i += 1
                    c_last_index += 1

        if mode == "one":
            while i < 5:
                if icm_argsort[a_last_index] not in top_5_indices:
                    top_5_indices.append(icm_argsort[a_last_index])
                    i += 1
                a_last_index += 1

                if ucm_argsort[b_last_index] not in top_5_indices:
                    top_5_indices.append(ucm_argsort[b_last_index])
                    i += 1
                b_last_index += 1

                if slim_argsort[c_last_index] not in top_5_indices:
                    top_5_indices.append(slim_argsort[c_last_index])
                    i += 1
                c_last_index += 1

        return top_5_indices

    def compute_avg_prediction(self, icm_prediction, ucm_prediction, slimBPR_prediction):
        icm_argsort = icm_prediction.argsort()[::-1]
        ucm_argsort = ucm_prediction.argsort()[::-1]
        slim_argsort = slimBPR_prediction.argsort()[::-1]

        avg_list = []
        top_5_indices = []

        for i in icm_argsort[:50]:
            sum = np.where(icm_argsort == i)[0][0] + \
                  np.where(ucm_argsort == i)[0][0] + \
                  np.where(slim_argsort == i)[0][0]
            avg = sum / 3
            avg_list.append(avg)

        avg_list_argsort = np.argsort(avg_list)
        for a in avg_list_argsort[:5]:
            top_5_indices.append(icm_argsort[a])

        return top_5_indices
