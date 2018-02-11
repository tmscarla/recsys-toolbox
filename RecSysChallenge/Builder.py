import numpy as np
import scipy as sc
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds
from pandas import DataFrame
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from tqdm import tqdm
import ast
from sklearn import feature_extraction
from sklearn import decomposition
from Utils import Utils

"""
This class contains fundamental functions to extract dataframes from the csv
and to build from them attributes and matrices.

The following matrices can be built up:
    - URM: User Rating Matrix
    - UCM: User Content Matrix
    - ICM: Item Content Matrix
"""


class Builder(object):

    def __init__(self):
        self.train_final = pd.read_csv('Data/train_final.csv', sep='\t')
        self.playlists_final = pd.read_csv('Data/playlists_final.csv', sep='\t')
        self.tracks_final = pd.read_csv('Data/tracks_final.csv', sep='\t')
        self.target_playlists = pd.read_csv('Data/target_playlists.csv', sep='\t')
        self.target_tracks = pd.read_csv('Data/target_tracks.csv', sep='\t')
        self.playlists = self.get_playlists()

    # GET DATAFRAMES

    def get_train_final(self):
        return self.train_final

    # GET ATTRIBUTES

    def unique_and_sorted(self, l):
        unique_l = list(set(l))
        unique_l.sort()
        return unique_l

    def get_tracks(self):
        tracks = self.tracks_final['track_id'].unique()
        return np.sort(tracks)

    def get_target_tracks(self):
        target_tracks = self.target_tracks['track_id'].unique()
        return np.sort(target_tracks)

    def get_playlists(self):
        playlists = self.train_final['playlist_id'].unique()
        return np.sort(playlists)

    def get_target_playlists(self):
        target_playlists = self.target_playlists['playlist_id'].unique()
        return np.sort(target_playlists)

    def get_artists(self):
        artists = [a for a in self.tracks_final['artist_id']]
        artists_unique = self.unique_and_sorted(artists)
        return artists_unique

    def get_albums(self):
        albums = [int(a.strip("[]"))
                  for a in self.tracks_final['album']
                  if a.strip("[]") != "None" and a.strip("[]") != ""]
        albums_unique = self.unique_and_sorted(albums)
        return albums_unique

    def get_tags(self):
        tags_df = self.tracks_final['tags'].apply(lambda x: ast.literal_eval(x))
        tags = [t for tags_list in tags_df for t in tags_list]
        tags_unique = self.unique_and_sorted(tags)
        return tags_unique

    # GET INDICES

    def get_nontarget_indices(self, target_tracks):
        tracks = self.get_tracks()
        return np.where(~(np.in1d(tracks, target_tracks)))

    def get_top_5_indices(self, URM_row, nontarget_indices, known_indices, owner_indices):
        URM_row[nontarget_indices] = 0
        URM_row[known_indices] = 0
        URM_row[owner_indices] = 0
        return np.flip(np.argsort(URM_row)[-5:], 0)

    def get_top_5_tracks_from_indices(self, indices):
        tracks = self.get_tracks()
        top_5_tracks = [tracks[i] for i in indices]
        return top_5_tracks

    def get_target_playlist_index(self, target_playlist):
        return np.where(self.playlists == target_playlist)[0][0]

    def get_all_tracks(self):
        tracks = [t for t in self.tracks_final['track_id']]
        tracks.sort()
        return tracks

    def get_all_playlists(self):
        playlists = self.train_final['playlist_id'].unique()
        return np.sort(playlists)

    def get_tracks_indices(self, tracks):
        return np.where(np.in1d(self.get_all_tracks(), tracks))[0]

    def get_playlists_indices(self, playlists):
        return np.where(np.in1d(self.get_all_playlists(), playlists))[0]

    # GET MATRICES

    def get_URM(self):
        print('Building URM...')

        grouped = self.train_final.groupby('playlist_id', as_index=True).apply(
            lambda x: list(x['track_id']))
        matrix = MultiLabelBinarizer(classes=self.get_tracks(), sparse_output=True).fit_transform(grouped)
        self.URM = matrix.tocsr()
        return self.URM

    def get_UCM(self, URM):
        print('Building UCM from URM...')

        UCM_tfidf = feature_extraction.text.TfidfTransformer().fit_transform(URM.T)
        UCM_tfidf = UCM_tfidf.T
        return UCM_tfidf

    def build_ICM(self):
        print('Building ICM from tracks_final...')

        # 1 - Artists
        artists_df = self.tracks_final.reindex(columns=['track_id', 'artist_id'])
        artists_df.sort_values(by='track_id', inplace=True)

        artists_list = [[a] for a in artists_df['artist_id']]
        icm_artists = MultiLabelBinarizer(classes=self.get_artists(), sparse_output=True).fit_transform(artists_list)
        icm_artists_csr = icm_artists.tocsr()

        # 2 - Albums
        albums_df = self.tracks_final.reindex(columns=['track_id', 'album'])
        albums_df.sort_values(by='track_id', inplace=True)

        albums_list = [ast.literal_eval(a)
                       if len(ast.literal_eval(a)) > 0 and ast.literal_eval(a)[0] is not None
                       else []
                       for a in albums_df['album']]
        icm_albums = MultiLabelBinarizer(classes=self.get_albums(), sparse_output=True).fit_transform(albums_list)
        icm_albums_csr = icm_albums.tocsr()

        # 3 - Tags
        tags_df = self.tracks_final.reindex(columns=['track_id', 'tags'])
        tags_df.sort_values(by='track_id', inplace=True)

        tags_list = [ast.literal_eval(t) for t in tags_df['tags']]
        icm_tags = MultiLabelBinarizer(classes=self.get_tags(), sparse_output=True).fit_transform(tags_list)
        icm_tags_csr = icm_tags.tocsr()

        # 4 - Stack together
        ICM = sparse.hstack((icm_artists_csr, icm_albums_csr, icm_tags_csr))

        # 5 - Tf-idf
        ICM_tfidf = feature_extraction.text.TfidfTransformer().fit_transform(ICM)
        ICM_tfidf = normalize(ICM_tfidf, axis=0, norm='l2')
        return ICM_tfidf.tocsr()


    def build_S_ICM_knn(self, ICM, knn):
        print("Building S from ICM with knn =", knn, "...")
        ICM_T = ICM.T
        ICM_T = ICM_T.tocsr()

        s_list = []

        for i in tqdm(range(0, ICM.shape[0])):
            s = ICM[i, :] * ICM_T
            r = s.data.argsort()[:-knn]
            s.data[r] = 0
            sparse.csr_matrix.eliminate_zeros(s)
            s_list.append(s)

        S = sparse.vstack(s_list)
        S.setdiag(0)
        return S


    def get_S_UCM_KNN(self, UCM, knn):
        print('Building S from UCM with knn =', knn, '...')

        S_matrix_list = []

        UCM = UCM.tocsr()
        UCM_T = UCM.T.tocsr()

        for i in tqdm(range(0, UCM.shape[1])):
            S_row = UCM_T[i] * UCM
            r = S_row.data.argsort()[:-knn]
            S_row.data[r] = 0

            sparse.csr_matrix.eliminate_zeros(S_row)
            S_matrix_list.append(S_row)

        S = sc.sparse.vstack(S_matrix_list)
        S.setdiag(0)

        return S

    # SVD - SINGOLAR VALUE DECOMPOSITION

    def get_S_ICM_SVD(self, ICM, k, knn):
        print('Computing S_ICM_SVD...')

        S_matrix_list = []

        u, s, vt = svds(ICM, k=k, which='LM')

        ut = u.T

        s_2_flatten = np.power(s, 2)
        s_2 = np.diagflat(s_2_flatten)
        s_2_csr = sparse.csr_matrix(s_2)

        for i in tqdm(range(0, u.shape[0])):
            S_row = u[i].dot(s_2_csr.dot(ut))
            r = S_row.argsort()[:-knn]
            S_row[r] = 0
            S_matrix_list.append(S_row)

        S = sc.sparse.vstack(S_matrix_list)

        return S

    def get_S_URM_SVD(self, URM, k, knn):
        print('Computing S _URM_SVD...')

        S_matrix_list = []

        URM = sparse.csr_matrix(URM, dtype=float)

        u, s, vt = svds(URM, k=k)
        v = vt.T

        for i in tqdm(range(0, v.shape[0])):
            S_row = v[i, :].dot(vt)
            r = S_row.argsort()[:-knn]
            S_row[r] = 0
            S_row_sparse = sparse.csr_matrix(S_row)
            sparse.csr_matrix.eliminate_zeros(S_row_sparse)
            S_matrix_list.append(S_row_sparse)

        S = sc.sparse.vstack(S_matrix_list)
        S.setdiag(0)

        return S

    def get_S_avg(self, S1, S2, a):

        S = (a * S1) + ((1-a) * S2)

        return S


print(Builder().build_ICM().shape)
