import numpy as np
import time
from tqdm import tqdm
import scipy as sc
from scipy import sparse

"""
Matrix Factorization with a Bayesian Personalized Ranking approach.
"""

class MFBPR(object):

    def __init__(self, URM,
                 n_factors=10,
                 learning_rate=0.1,
                 epochs=10,
                 user_regularization=0.1,
                 positive_item_regularization=0.1,
                 negative_item_regularization=0.1):
        self.URM = URM
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.user_regularization = user_regularization
        self.positive_item_regularization = positive_item_regularization
        self.negative_item_regularization = negative_item_regularization

        self.n_users = self.URM.shape[0]
        self.n_items = self.URM.shape[1]
        self.user_factors = np.random.random_sample((self.n_users, n_factors))
        self.item_factors = np.random.random_sample((self.n_items, n_factors))

    def sampleTriplet(self):

        # By randomly selecting a user in this way we could end up
        # with a user with no interactions
        # user_id = np.random.randint(0, n_users)

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
        numPositiveIteractions = int(self.URM.nnz * 0.1)

        # Uniform user sampling without replacement
        for it in tqdm(range(numPositiveIteractions)):
            u, i, j = self.sampleTriplet()
            self.update_factors(u, i, j)

    def update_factors(self, u, i, j, update_u=True, update_i=True, update_j=True):
        """Apply SGD update"""

        x = np.dot(self.user_factors[u, :], self.item_factors[i, :] - self.item_factors[j, :])

        z = 1.0 / (1.0 + np.exp(x))

        if update_u:
            d = (self.item_factors[i, :] - self.item_factors[j, :]) * z \
                - self.user_regularization * self.user_factors[u, :]
            self.user_factors[u, :] += self.learning_rate * d
        if update_i:
            d = self.user_factors[u, :] * z - self.positive_item_regularization * self.item_factors[i, :]
            self.item_factors[i, :] += self.learning_rate * d
        if update_j:
            d = -self.user_factors[u, :] * z - self.negative_item_regularization * self.item_factors[j, :]
            self.item_factors[j, :] += self.learning_rate * d

    def fit(self):
        print('Fitting MFBPR...')

        for numEpoch in range(self.epochs):
            print('Epoch:', numEpoch)
            self.epochIteration()

    def predict(self, u):
        rec = np.dot(self.user_factors[u], self.item_factors.T)
        rec_csr = sparse.csr_matrix(rec)
        return rec_csr

