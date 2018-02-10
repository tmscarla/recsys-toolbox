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
from sklearn.preprocessing import normalize

from FM import FactorizationMachine


class Recommender(object):

    def __init__(self):
        self.URM = None
        self.target_playlists = None
        self.target_tracks = None
        self.k = None
        self.S_ICM = None
        self.S_UCM = None
        self.S_URM = None
        self.Slim = None
        self.mfbpr = None
        self.a = 0.25
        self.b = 0.25
        self.c = 0.25
        self.d = 0.25
        self.e = 0.25
        self.is_test = False

    def fit(self, URM_train, target_playlists, target_tracks, num_playlist_to_recommend,
            S_ICM, S_UCM, S_URM, Slim, mfbpr,
            a, b, c, d, e,
            is_test):
        self.URM = URM_train
        self.target_playlists = target_playlists
        self.target_tracks = target_tracks
        self.num_playlist_to_recommend = num_playlist_to_recommend
        self.S_ICM = S_ICM
        self.S_UCM = S_UCM
        self.S_URM = S_URM
        self.Slim = Slim
        self.mfbpr = mfbpr
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.is_test = is_test

    def recommend(self):
        b = Builder()
        #nb = NewBuilder()

        nontarget_indices = b.get_nontarget_indices(self.target_tracks)

        URM_T = self.URM.T
        URM_tfidf_T = feature_extraction.text.TfidfTransformer().fit_transform(URM_T)
        URM_tfidf = URM_tfidf_T.T
        URM_tfidf_csr = URM_tfidf.tocsr()

        dataframe_list = []

        print('Predicting...', flush=True)
        for i in tqdm(range(0, self.num_playlist_to_recommend)):
            # Iterate over indices of target playlists
            index = b.get_target_playlist_index(self.target_playlists[i])

            # Compute relevant indices for the prediction
            known_indices = np.nonzero(self.URM[index].toarray().flatten())
            #owner_tracks = nb.get_tracks_from_playlist_owner(self.target_playlists[i])
            #owner_indices = nb.get_tracks_indices(owner_tracks)
            owner_indices = []

            # Calculate recommenders contributions
            icm_prediction = URM_tfidf_csr[index, :] * self.S_ICM
            icm_prediction = normalize(icm_prediction, axis=1, norm='l2')

            #ucm_prediction = URM_tfidf_csr[index, :] * self.S_UCM
            #ucm_prediction = normalize(ucm_prediction, axis=1, norm='l2')
            #
            #svd_prediction = self.URM[index, :] * self.S_URM
            #svd_prediction = normalize(svd_prediction, axis=1, norm='l2')
            #
            #slimBPR_prediction = URM_tfidf_csr[index, :] * self.Slim
            #slimBPR_prediction = normalize(slimBPR_prediction, axis=1, norm='l2')

            # MFBPR_prediction = self.mfbpr.predict(index)
            # MFBPR_prediction = normalize(MFBPR_prediction, axis=1, norm='l2')

            # Weighted average of recommendations
            # URM_row = self.a * icm_prediction +\
            #           self.b * ucm_prediction + \
            URM_row = self.d * icm_prediction
            #           self.c * svd_prediction
            #           self.e * MFBPR_prediction

            #URM_row = icm_prediction

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

    def recommend_avg_similarity(self, avg, beta):
        b = Builder()
        #2nb = NewBuilder()

        nontarget_indices = b.get_nontarget_indices(self.target_tracks)

        URM_T = self.URM.T
        URM_tfidf_T = feature_extraction.text.TfidfTransformer().fit_transform(URM_T)
        URM_tfidf = URM_tfidf_T.T
        URM_tfidf_csr = URM_tfidf.tocsr()

        dataframe_list = []

        # Weighted average of S_ICM and S_UCM
        S_avg = (avg * self.S_ICM) + ((1-avg) * self.S_UCM)

        print('Predicting avg_similarity...', flush=True)
        for i in tqdm(range(0, self.num_playlist_to_recommend)):
            # Iterate over indices of target playlists
            index = b.get_target_playlist_index(self.target_playlists[i])

            # Compute relevant indices for the prediction
            known_indices = np.nonzero(self.URM[index].toarray().flatten())
            #owner_tracks = nb.get_tracks_from_playlist_owner(self.target_playlists[i])
            #owner_indices = nb.get_tracks_indices(owner_tracks)
            owner_indices = []

            # Calculate recommenders contributions
            avg_prediction = URM_tfidf_csr[index, :] * S_avg
            #avg_prediction = normalize(avg_prediction, axis=1, norm='l2')

            #slimBPR_prediction = URM_tfidf_csr[index, :] * self.Slim
            #slimBPR_prediction = normalize(slimBPR_prediction, axis=1, norm='l2')

            # Weighted average of recommendations
            URM_row = (beta * avg_prediction) #+ ((1-beta) * slimBPR_prediction)

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

    def recommend_round_robin(self, mode, a, b, c):
        builder = Builder()
        nontarget_indices = builder.get_nontarget_indices(self.target_tracks)

        URM_T = self.URM.T
        URM_tfidf_T = feature_extraction.text.TfidfTransformer().fit_transform(URM_T)
        URM_tfidf = URM_tfidf_T.T
        URM_tfidf_csr = URM_tfidf.tocsr()

        dataframe_list = []

        print('Predicting round_robin with mode =', mode, '...', flush=True)
        for i in tqdm(range(0, self.num_playlist_to_recommend)):
            # Iterate over indices of target playlists
            index = builder.get_target_playlist_index(self.target_playlists[i])

            # Compute relevant indices for the prediction
            known_indices = np.nonzero(self.URM[index].toarray().flatten())
            #owner_tracks = nb.get_tracks_from_playlist_owner(self.target_playlists[i])
            #owner_indices = nb.get_tracks_indices(owner_tracks)
            owner_indices = []

            # Calculate recommenders contributions
            icm_prediction = self.URM[index, :] * self.S_ICM
            icm_prediction_flatten = icm_prediction.toarray().flatten()
            icm_prediction_flatten[known_indices] = 0
            icm_prediction_flatten[nontarget_indices] = 0
            # icm_prediction = normalize(icm_prediction, axis=1, norm='l2')

            ucm_prediction = self.URM[index, :] * self.S_UCM
            ucm_prediction_flatten = ucm_prediction.toarray().flatten()
            ucm_prediction_flatten[known_indices] = 0
            ucm_prediction_flatten[nontarget_indices] = 0
            # ucm_prediction = normalize(ucm_prediction, axis=1, norm='l2')

            slimBPR_prediction = URM_tfidf_csr[index, :] * self.Slim
            slimBPR_prediction_flatten = slimBPR_prediction.toarray().flatten()
            slimBPR_prediction_flatten[known_indices] = 0
            slimBPR_prediction_flatten[nontarget_indices] = 0
            #slimBPR_prediction = normalize(slimBPR_prediction, axis=1, norm='l2')

            # Round Robin prediction
            top_5_indices = self.round_robin(icm_prediction_flatten,
                                             ucm_prediction_flatten,
                                             slimBPR_prediction_flatten,
                                             mode, a, b, c)
            top_5_tracks = builder.get_top_5_tracks_from_indices(top_5_indices)
            top_5_tracks_string = ' '.join([str(i) for i in top_5_tracks])

            # Create dataset
            if self.is_test:
                dataframe_list.append([self.target_playlists[i], top_5_tracks])
            else:
                dataframe_list.append([self.target_playlists[i], top_5_tracks_string])

        dataframe = pd.DataFrame(dataframe_list, columns=['playlist_id', 'track_ids'])

        return dataframe

    def recommend_avg_prediction(self):
        builder = Builder()
        # nb = NewBuilder()

        nontarget_indices = builder.get_nontarget_indices(self.target_tracks)

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
            # owner_tracks = nb.get_tracks_from_playlist_owner(self.target_playlists[i])
            # owner_indices = nb.get_tracks_indices(owner_tracks)
            owner_indices = []

            # Calculate recommenders contributions
            icm_prediction = self.URM[index, :] * self.S_ICM
            icm_prediction_flatten = icm_prediction.toarray().flatten()
            icm_prediction_flatten[known_indices] = 0
            icm_prediction_flatten[nontarget_indices] = 0
            # icm_prediction = normalize(icm_prediction, axis=1, norm='l2')

            ucm_prediction = self.URM[index, :] * self.S_UCM
            ucm_prediction_flatten = ucm_prediction.toarray().flatten()
            ucm_prediction_flatten[known_indices] = 0
            ucm_prediction_flatten[nontarget_indices] = 0
            # ucm_prediction = normalize(ucm_prediction, axis=1, norm='l2')

            slimBPR_prediction = URM_tfidf_csr[index, :] * self.Slim
            slimBPR_prediction_flatten = slimBPR_prediction.toarray().flatten()
            slimBPR_prediction_flatten[known_indices] = 0
            slimBPR_prediction_flatten[nontarget_indices] = 0
            #slimBPR_prediction = normalize(slimBPR_prediction, axis=1, norm='l2')

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

    def recommend_fm(self, ICM):
        builder = Builder()
        newBuilder = NewBuilder()
        fm = FactorizationMachine()
        fm.fit(self.URM, ICM, self.target_playlists, self.target_tracks)
        target_tracks_i = newBuilder.get_tracks_indices(self.target_tracks)

        r = fm.recommend()
        print(r)
        print(r.shape)

        nontarget_indices = builder.get_nontarget_indices(self.target_tracks)
        dataframe_list = []
        offset = 0

        print('Predicting FM...', flush=True)
        for i in tqdm(range(0, self.num_playlist_to_recommend)):
            # Iterate over indices of target playlists
            index = builder.get_target_playlist_index(self.target_playlists[i])
            owner_indices = []
            known_indices = np.nonzero(self.URM[index].toarray().flatten())

            # Increment offset of prediction array r
            if i != 0 and i % fm.n_tracks == 0:
                offset += fm.n_tracks

            # Weighted average of recommendations
            URM_row_flatten = np.zeros(fm.n_tracks)
            for t_i in target_tracks_i:
                URM_row_flatten[t_i] = r[i + offset]

            # Make prediction
            top_5_indices = builder.get_top_5_indices(URM_row_flatten, nontarget_indices, known_indices, owner_indices)
            top_5_tracks = builder.get_top_5_tracks_from_indices(top_5_indices)
            top_5_tracks_string = ' '.join([str(i) for i in top_5_tracks])

            # Create dataset
            if self.is_test:
                dataframe_list.append([self.target_playlists[i], top_5_tracks])
            else:
                dataframe_list.append([self.target_playlists[i], top_5_tracks_string])

        dataframe = pd.DataFrame(dataframe_list, columns=['playlist_id', 'track_ids'])

        return dataframe



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
            sum = np.where(icm_argsort == i)[0][0] +\
                  np.where(ucm_argsort == i)[0][0] +\
                  np.where(slim_argsort == i)[0][0]
            avg = sum / 3
            avg_list.append(avg)

        avg_list_argsort = np.argsort(avg_list)
        for a in avg_list_argsort[:5]:
            top_5_indices.append(icm_argsort[a])

        return top_5_indices
