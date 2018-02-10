from Builder import Builder
from Recommenders import Recommender as isr
from Evaluator import Evaluator
from SlimBPR import SlimBPR
import pickle
import ast
import pandas as pd
from scipy import sparse
import numpy as np
from tqdm import tqdm
from SlimBPR import SlimBPR
from MFBPR import MFBPR
from fastFM import sgd
from fastFM import bpr



class FactorizationMachine(object):
    """
    URM: (45.649, 100.000) nnz: 1.040.522
    ICM: (100.000, 77.040)

    train_length = 222.689

    train_shape: (2.081.044, 222.689)
    """

    def __init__(self, n_iter=100,
                 init_stdev=0.1,
                 rank=8,
                 random_state=123,
                 l2_reg_w=0.1,
                 l2_reg_V=0.1,
                 l2_reg=0,
                 step_size=0.1):

        self.fastFM = bpr.FMRecommender(n_iter=n_iter,
                                        init_stdev=init_stdev,
                                        rank=rank,
                                        random_state=random_state,
                                        l2_reg_w=l2_reg_w,
                                        l2_reg_V=l2_reg_V,
                                        l2_reg=l2_reg,
                                        step_size=step_size)
        self.URM = None
        self.ICM = None
        self.train_x = None
        self.train_y = None
        self.test_playlists = None
        self.test_tracks = None
        self.test_x = None
        self.train_length = 222689
        self.non_zeros = 0
        self.n_playlists = 45649
        self.n_tracks = 100000
        self.n_attributes = 77040

    def fit(self, URM, ICM, test_playlists, test_tracks):
        self.URM = URM
        self.ICM = ICM
        self.non_zeros = self.URM.getnnz()
        self.test_playlists = test_playlists
        self.test_tracks = test_tracks

        self.build_test_x()
        self.build_train_x()
        self.build_train_y()

        self.fastFM.fit(self.train_x, self.train_y)

    def build_train_x(self):
        print("Build train_x...")
        URM_coo = self.URM.tocoo()
        URM_lil = self.URM.tolil()
        ICM_lil = self.ICM.tolil()

        rows_list = []

        print("Build rows for positive values...")
        for i in tqdm(range(0, self.non_zeros)):
            # Identifies playlists, tracks and attributes of the non-zeros
            p_i = URM_coo.row[i]
            t_i = URM_coo.col[i]

            self.build_rows(p_i, t_i, ICM_lil, rows_list, False)

        print("Build rows for negative values...")
        for i in tqdm(range(0, self.non_zeros)):
            p_i = np.random.choice(self.n_playlists, 1)[0]

            neg_track_selected = False
            tracks_in_playlist = URM_lil[p_i].rows[0]

            while (not neg_track_selected):
                t_i = np.random.randint(0, self.n_tracks)

                if (t_i not in tracks_in_playlist):
                    neg_track_selected = True

            self.build_rows(p_i, t_i, ICM_lil, rows_list, False)

        print("Stack all the rows together...")
        # Stack all the rows together
        self.train_x = sparse.vstack(rows_list)
        print(self.train_x.shape)

    def build_train_y(self):
        print("Build train_y...")
        positives = np.random.choice(self.non_zeros, self.non_zeros, replace=False)
        negatives = np.random.choice(self.non_zeros, self.non_zeros, replace=False)

        self.train_y = np.stack((positives, negatives), axis=-1)
        print(self.train_y.shape)

    def build_rows(self, p_i, t_i, ICM_lil, rows_list, is_test):

        # Playlists rows
        playlists = sparse.coo_matrix(([1], ([0], [p_i])), shape=(1, self.n_playlists))
        playlists_csr = playlists.tocsr()

        icm_rows = []

        # True when building the test
        if is_test:

            # Tracks test rows
            row = np.zeros(len(t_i), dtype=np.int)
            col = np.array(t_i)
            data = np.ones(len(t_i), dtype=np.int)
            tracks_csr = sparse.csr_matrix((data, (row, col)), shape=(1, self.n_tracks))

            # Attributes test rows
            for t in t_i:
                icm_rows.append(self.ICM[t])

            icm_stack = sparse.vstack(icm_rows)
            attrs_values = np.squeeze(np.asarray(icm_stack.sum(axis=0)))
            nonzero_mask = np.where(attrs_values > 0)
            attrs_values_nnz = attrs_values[nonzero_mask]

            row = np.zeros(len(attrs_values_nnz), dtype=np.int)
            col = np.nonzero(attrs_values)[0]
            data = attrs_values_nnz

            attributes_csr = sparse.csr_matrix((data, (row, col)), shape=(1, self.n_attributes))

        # In case of train_x and train_y
        else:
            tracks_csr = sparse.csr_matrix(([1], ([0], [t_i])), shape=(1, self.n_tracks))
            attributes_csr = ICM_lil[t_i].tocsr()

        # Stack them together
        row = sparse.hstack((playlists_csr, tracks_csr, attributes_csr))
        row_csr = row.tocsr()
        rows_list.append(row_csr)

    def build_test_x(self):
        print("Build test_x...")
        b = Builder()
        playlists_indices = b.get_playlists_indices(self.test_playlists)
        tracks_indices = b.get_tracks_indices(self.test_tracks)
        ICM_lil = self.ICM.tolil()
        URM_lil = self.URM.tolil()

        rows_list = []

        for p_i in tqdm(playlists_indices):
            p_i_tracks = URM_lil[p_i].rows[0]
            self.build_rows(p_i, p_i_tracks, ICM_lil, rows_list, True)

        # Stack all the rows together
        self.test_x = sparse.vstack(rows_list)

    def recommend(self):
        return self.fastFM.predict(self.test_x)
