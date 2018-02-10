import numpy as np
import time
from tqdm import tqdm
import scipy as sc
from scipy import sparse
from scipy.sparse.linalg import norm
from Builder import Builder
from Utils import Utils
from sklearn import feature_extraction

class SlimBPR(object):
    """ SLIM_BPR recommender with cosine similarity and no shrinkage"""

    def __init__(self, URM,
                 learning_rate=0.01,
                 epochs=1,
                 positive_item_regularization=1.0,
                 negative_item_regularization=1.0,
                 nnz = 1):
        self.URM = URM
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.positive_item_regularization = positive_item_regularization
        self.negative_item_regularization = negative_item_regularization
        self.nnz = nnz
        self.n_users = self.URM.shape[0]
        self.n_items = self.URM.shape[1]

        # Use lil_matrix to incrementally build the similarity matrix
        self.similarity_matrix = sparse.lil_matrix((self.n_items, self.n_items))


    def sampleTriplet(self):

        # By randomly selecting a user in this way we could end up
        # with a user with no interactions
        # user_id = np.random.randint(0, self.n_users)

        user_id = np.random.choice(self.n_users)

        # Get user seen items and choose one
        userSeenItems = self.URM[user_id, :].indices
        pos_item_id = np.random.choice(userSeenItems)

        negItemSelected = False

        # It's faster to just try again then to build a mapping of the non-seen items
        while (not negItemSelected):
            neg_item_id = np.random.randint(0, self.n_items)

            if (neg_item_id not in userSeenItems):
                negItemSelected = True

        return user_id, pos_item_id, neg_item_id

    def epochIteration(self):

        # Get number of available interactions
        numPositiveIteractions = int(self.URM.nnz * self.nnz)

        # Uniform user sampling without replacement
        for num_sample in tqdm(range(numPositiveIteractions)):

            # Sample
            user_id, positive_item_id, negative_item_id = self.sampleTriplet()

            userSeenItems = self.URM[user_id, :].indices

            # Prediction
            x_i = self.similarity_matrix[positive_item_id, userSeenItems].sum()
            x_j = self.similarity_matrix[negative_item_id, userSeenItems].sum()

            # Gradient
            x_ij = x_i - x_j

            gradient = 1 / (1 + np.exp(x_ij))

            for i in userSeenItems:
                dp = gradient - self.positive_item_regularization * x_i
                self.similarity_matrix[positive_item_id, i] = self.similarity_matrix[positive_item_id, i] +\
                    self.learning_rate * dp
                dn = gradient - self.negative_item_regularization * x_j
                self.similarity_matrix[negative_item_id, i] = self.similarity_matrix[negative_item_id, i] -\
                    self.learning_rate * dn

            self.similarity_matrix[positive_item_id, positive_item_id] = 0
            self.similarity_matrix[negative_item_id, negative_item_id] = 0

    def get_S_SLIM_BPR(self, knn):
        print('Get S SLIM BPR...')

        for numEpoch in range(self.epochs):
            print('Epoch:', numEpoch)
            self.epochIteration()

        print('Keeping only knn =', knn, '...')
        similarity_matrix_csr = self.similarity_matrix.tocsr()

        for r in tqdm(range(0, similarity_matrix_csr.shape[0])):
            indices = similarity_matrix_csr[r, :].data.argsort()[:-knn]
            similarity_matrix_csr[r, :].data[indices] = 0
        sparse.csr_matrix.eliminate_zeros(similarity_matrix_csr)

        return similarity_matrix_csr
