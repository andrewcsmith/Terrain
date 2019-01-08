import harmonic_distance as hd
import numpy as np
import tensorflow as tf

class Optimizer:
    def __init__(self, c=0.02, learning_rate=1.0e-4, 
                convergence_threshold=1.0e-8, max_iters=10000,
                n_points=1, session=None):
        self.__init_session(session)
        # Initialize relevant parameters
        self.c = c
        self.learning_rate = learning_rate
        self.convergence_threshold = convergence_threshold
        self.max_iters = max_iters
        # Hard-code dimensions to 2 for now
        self.dimensions = 2
        self.offset = 0.0
        with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
            self.log_pitches = tf.get_variable(f"log_pitches_{n_points}x{self.dimensions}", [n_points, self.dimensions], dtype=tf.float64)
        # Run the initializer, b/c we might as well
        self.session.run(tf.global_variables_initializer())
    
    def __del__(self):
        print("Ending session")
        self.session.close()

    def __init_session(self, session):
        if session is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.session = tf.Session(config=config)
        else: 
            self.session = session
    
    def populate_vectors(self, prime_limits, bounds, hd_threshold=9.0):
        vectors = hd.vectors.space_graph_altered_permutations(prime_limits, bounds=bounds)
        hds = hd.tenney.hd_aggregate_graph(tf.cast(vectors[:, None, :], tf.float64))
        self.vectors = tf.boolean_mask(vectors, hds < hd_threshold)
    
    def populate_perms(self):
        perms = hd.cartesian.permutations(self.vectors, times=2)
        np_perms = self.session.run(perms)
        self.perms = tf.constant(np_perms, dtype=tf.float64)

    def populate_distances(self):
        hds = hd.tenney.hd_aggregate_graph(self.perms) + 1.0
        pds = hd.tenney.pd_aggregate_graph(self.perms)
        hds_np, pds_np = self.session.run([hds, pds])
        self.hds = tf.constant(hds_np)
        self.pds = tf.constant(pds_np)
    
    def populate_loss(self):
        self.loss = hd.optimize.parabolic_loss_function(self.pds, self.hds, self.log_pitches, a=self.c, b=self.c)
    
    def get_stopping_op(self):
        return hd.optimize.stopping_op(self.loss, [self.log_pitches])
    
    def optimize(self, starting_pitches):
        starting = starting_pitches + self.offset
        self.session.run(self.log_pitches.assign(starting))
        stopping_op = self.get_stopping_op()
        for idx in range(self.max_iters):
            if (self.session.run(stopping_op)):
                print("Converged at iteration: ", idx)
                self.ending_pitches = np.array(self.session.run(self.log_pitches))
                return self.ending_pitches
        print("Did not converge.")

class DownhillOptimizer(Optimizer):
    def populate_vectors(self, prime_limits, bounds, hd_threshold=9.0):
        """
        Mask the vectors Tensor to only include root_valences that are downhill
        """
        super().populate_vectors(prime_limits, bounds, hd_threshold=hd_threshold)
        root_valences = hd.tenney.hd_root_valence(self.vectors)
        downhill = tf.boolean_mask(self.vectors, root_valences < 0.0)
        self.vectors = downhill

    def populate_perms(self):
        """
        For this 1-d optimizer, we just want to set the second point to 1/1
        """
        self.perms = tf.stack([tf.zeros_like(self.vectors), self.vectors], 1)
