import numpy as np
import pandas as pd
from os.path import dirname, abspath

"""
Top Popular recommender.
Suggests tracks according to their global popularity,
thus the number of playlists that contain them.

Return a dataframe in the submission format.
"""


class TopPopRec(object):

    def fit(self):
        # Get the parent directory path
        directory = dirname(dirname(abspath(__file__)))

        # Load csv
        self.train_final = pd.read_csv(directory + '/Data/train_final.csv', sep='\t')
        self.target_playlists = np.array(pd.read_csv(directory + '/Data/target_playlists.csv', sep='\t'))

    def recommend(self):
        # Read .csv file
        df = np.array(self.train_final)

        # Count tracks popularity
        unique, counts = np.unique(df[:,[1]], return_counts=True)
        p = np.asarray((unique, counts)).T
        s = p[p[:, 1].argsort()]

        m, n = np.shape(s)
        top5 = s[m-5:m,:]
        top5_ordered = top5[:,:1]
        top5_final = top5_ordered[::-1]
        top = np.reshape(top5_final, (1,5))
        a = ' '.join(map(str, top[0]))


        # Create an empty dataset to handle the recommendations
        dataframe = pd.DataFrame(index=range(0, 10000), columns=['playlist_id', 'track_ids'], dtype=int)
        for index, row in dataframe.iterrows():
            row['playlist_id'] = int(self.target_playlists[index])
            row['track_ids'] = a

        return dataframe
