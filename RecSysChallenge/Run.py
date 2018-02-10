import Recsys as rs

"""
From here you can make prediction or test each recommender.
Just uncomment one of the following line and run.

Hyperparameters for each recommender can be changed in the
Recsys.py file.
"""

rs.hybrid_rec(is_test=True)
# rs.top_pop_rec()
# rs.item_based(is_test=True)
# rs.round_robin_rec(is_test=True, avg_mode=False)
# rs.round_robin_rec(is_test=True, avg_mode=True)
# rs.item_based(is_test=True)
# rs.SVD(is_test=True)
# rs.item_user_avg(is_test=True)
# rs.collaborative_filtering(is_test=True)

