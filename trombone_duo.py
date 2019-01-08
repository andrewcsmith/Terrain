import harmonic_distance as hd
import numpy as np
import tensorflow as tf
from optimizer import Optimizer

class TromboneDuo:
    def __init__(self, c=0.02, learning_rate=1.0e-4, convergence_threshold=1.0e-8, max_iters=10000):
        self.optimizer = Optimizer(c=c, learning_rate=learning_rate, 
            convergence_threshold=convergence_threshold, 
            max_iters=max_iters, n_points=1)
    
    def cache_vectors(self, prime_limits, bounds, hd_threshold=9.0):
        self.optimizer.populate_vectors(prime_limits, bounds, hd_threshold=hd_threshold)
        self.optimizer.populate_perms()
        self.optimizer.populate_distances()
        self.optimizer.populate_loss()
    
    def find_new_pitches(self):
        self.current_pitches = self.optimizer.optimize(self.current_pitches)
        return self.current_pitches
    
    def current_ratio(self):
        diffs = self.optimizer.pds - self.current_pitches
        diffs = tf.abs(diffs)
        diffs = tf.reduce_sum(diffs, axis=-1)
        idx = tf.argmin(diffs, axis=0)
        vecs = self.optimizer.perms[idx, :, :]
        return self.optimizer.session.run(vecs)
    