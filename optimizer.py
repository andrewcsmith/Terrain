import harmonic_distance as hd
import numpy as np
import tensorflow as tf

class Optimizer:
    """
    Base class to handle all optimizations of a given collection of points.
    """
    def __init__(self, c=0.05, learning_rate=1.0e-4, 
                convergence_threshold=1.0e-8, max_iters=10000,
                n_points=1, session=None, dimensions=2):
        self.__init_session(session)
        # Initialize relevant parameters
        self.c = c
        self.learning_rate = learning_rate
        self.convergence_threshold = convergence_threshold
        self.max_iters = max_iters
        # Hard-code dimensions to 2 for now
        self.dimensions = dimensions
        self.offset = 0.0
        self.n_points = n_points
        self.log_pitches_name = f"log_pitches_{self.n_points}x{self.dimensions}"
        with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
            self.log_pitches = tf.get_variable(self.log_pitches_name, [self.n_points, self.dimensions], dtype=tf.float64)
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
        perms = hd.cartesian.permutations(self.vectors, times=self.dimensions)
        np_perms = self.session.run(perms)
        self.load_perms(np_perms)

    def populate_distances(self):
        hds = hd.tenney.hd_aggregate_graph(self.perms) + 1.0
        pds = hd.tenney.pd_aggregate_graph(self.perms)
        np_hds, np_pds = self.session.run([hds, pds])
        self.load_hds(np_hds)
        self.load_pds(np_pds)

    def load_perms(self, np_perms):
        self.perms = tf.constant(np_perms, dtype=tf.float64)

    def load_hds(self, np_hds):
        self.hds = tf.constant(np_hds, dtype=tf.float64)
    
    def load_pds(self, np_pds):
        self.pds = tf.constant(np_pds, dtype=tf.float64)
    
    def populate_loss(self):
        curves = tf.constant(self.c, shape=self.log_pitches.shape[1:], dtype=tf.float64)
        self.loss = hd.optimize.parabolic_loss_function(self.pds, self.hds, self.log_pitches, curves=curves)
    
    def get_stopping_op(self):
        return hd.optimize.stopping_op(self.loss, [self.log_pitches], lr=self.learning_rate, ct=self.convergence_threshold)
        
    def assign_starting_pitches(self, starting_pitches):
        starting = starting_pitches + self.offset
        with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
            self.session.run(tf.get_variable(self.log_pitches_name, dtype=tf.float64).assign(starting))
    
    def optimize(self, starting_pitches):
        self.assign_starting_pitches(starting_pitches)
        stopping_op = self.get_stopping_op()
        for idx in range(self.max_iters):
            if (self.session.run(stopping_op)):
                print("Converged at iteration: ", idx)
                self.ending_pitches = np.array(self.session.run(self.log_pitches))
                return self.ending_pitches
        print("Did not converge.")

class BoundedOptimizer(Optimizer):
    """
    Optimizes all points within given boundaries. For the purposes of the piece
    "Terrain," the bounds are consistently the edges of the trombones' ranges
    on their slides over a given partial. This ensures that the trombones are
    not asked to gliss to a point that is unavailable with their current slide positions.
    """
    def __set_2d_bounds(self, bounds):
        lower_bound = self.pds >= bounds[:, 0]
        upper_bound = self.pds <= bounds[:, 1]
        mask = tf.reduce_all(tf.logical_and(lower_bound, upper_bound), axis=1)
        self.masked_pds = tf.boolean_mask(self.pds, mask)
        self.masked_hds = tf.boolean_mask(self.hds, mask)
    
    def populate_loss(self, bounds):
        self.__set_2d_bounds(bounds)
        curves = tf.constant(self.c, shape=self.log_pitches.shape[1:], dtype=tf.float64)
        self.loss = hd.optimize.parabolic_loss_function(self.masked_pds, 
            self.masked_hds, self.log_pitches, curves=curves)

class LastDimensionOptimizer(Optimizer):
    """
    Given a handful of points, optimizes only the final dimension. This finds
    local minima along a given axis. For the purposes of the piece "Terrain,"
    this function is used to create a scale that harmonizes with an underlying chord.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
            self.var_list = tf.get_variable(f"var_list_{self.n_points}x1", [self.n_points, 1], dtype=tf.float64)
            self.stable_pitches = tf.get_variable(f"stable_pitches_{self.n_points}x{self.dimensions-1}", [self.n_points, self.dimensions-1], dtype=tf.float64)
            self.log_pitches = tf.concat([self.stable_pitches, self.var_list], -1)
            
    def populate_loss(self):
        curves = tf.constant(self.c, shape=self.log_pitches.shape[1:], dtype=tf.float64)
        self.loss = hd.optimize.parabolic_loss_function(self.pds, self.hds, self.log_pitches, curves=curves)            
        
    def get_stopping_op(self):
        return hd.optimize.stopping_op(self.loss, self.var_list, 
            lr=self.learning_rate, ct=self.convergence_threshold)
    
    def assign_starting_pitches(self, starting_pitches):
        start = starting_pitches + self.offset
        self.session.run(self.stable_pitches.assign(start[:, :-1]))
        self.session.run(self.var_list.assign(start[:, -1:]))
        
class DownhillOptimizer(Optimizer):
    """
    This optimizer only allows for pitches that are "downhill" from the current
    base root pitch, given Tenney's notion of a downhill harmony (from the
    article "About Changes, 64 studies for 6 harps").
    """
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
        self.perms = self.vectors[:, None, :]
