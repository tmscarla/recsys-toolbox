from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import shuffle
from scipy import sparse
from time import time
import pandas as pd
import numpy as np
from tqdm import *
from Builder import Builder

class Evaluator(object):

    def __init__(self):
        self.b = Builder()
        self.URM_train = None
        self.test_df = None
        self.target_playlists = None
        self.target_tracks = None
        self.num_playlists_to_test = 10000

    def get_URM_train(self):
        return self.URM_train

    def get_test_df(self):
        return self.test_df

    def get_target_playlists(self):
        return self.target_playlists

    def get_target_tracks(self):
        return self.target_tracks

    def split(self):
        """
        Splits the dataset into training and test set.
        Builds the URM train csr matrix and the test dataframe in a
        submission-like structure.
        """

        print('Splitting the dataset...')

        # Load the original data set and group by playlist

        URM_df = self.b.get_train_final()
        grouped = URM_df.groupby('playlist_id', as_index=True).apply(
            lambda x: list(x['track_id']))
        grouped.sort_index(inplace=True)

        # Set num_playlist_to_test

        self.num_playlists_to_test = int(self.b.get_URM().shape[0] * 0.20)

        # Find indices of playlists to test and set target_playlists

        testable_idx = grouped[[len(x) >= 10 for x in grouped]].index
        test_idx = np.random.choice(testable_idx, self.num_playlists_to_test, replace=False)
        test_idx.sort()
        self.target_playlists = test_idx

        # Extract the test set portion of the data set

        test_mask = grouped[test_idx]
        test_mask.sort_index(inplace=True)

        # Iterate over the test set to randomly remove 5 tracks from each playlist

        test_df_list = []
        i = 0
        for t in test_mask:
            t_tracks_to_test = np.random.choice(t, 5, replace=False)
            test_df_list.append([test_idx[i], t_tracks_to_test])
            for tt in t_tracks_to_test:
                t.remove(tt)
            i += 1

        # Build test_df and URM_train

        self.test_df = pd.DataFrame(test_df_list, columns=['playlist_id', 'track_ids'])

        URM_train_matrix = MultiLabelBinarizer(classes=self.b.get_tracks(), sparse_output=True).fit_transform(grouped)
        self.URM_train = URM_train_matrix.tocsr()

        # Set target tracks

        t_list = [t for sub in self.test_df['track_ids'] for t in sub]
        t_list_unique = list(set(t_list))
        t_list_unique.sort()

        self.target_tracks = t_list_unique

    def ap(self, recommended_items, relevant_items):
        """
        Compute AP = Average Precision
        """

        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

        # Cumulative sum: precision at 1, at 2, at 3 ...
        p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

        map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

        return map_score

    def map5(self, train_df):
        """
        Compute MAP@5 on train_dataframe with known
        results in the test_dataframe
        """
        map5 = 0

        train_matrix = pd.DataFrame.as_matrix(train_df['track_ids'])
        test_matrix = pd.DataFrame.as_matrix(self.test_df['track_ids'])

        for i in range(0, self.num_playlists_to_test):
            map5 = map5 + self.ap(train_matrix[i], test_matrix[i])

        return map5/self.num_playlists_to_test
