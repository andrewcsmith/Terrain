import harmonic_distance as hd
import tensorflow as tf
import numpy as np
import yaml

from optimizer import *
from trombone_duo import *

F = np.log2(4.0 / 3.0)
G = np.log2(3.0 / 2.0)
A = np.log2(27.0 / 16.0)

class Structure:
    def __init__(self):
        self.downhill_optimizer = self.__init_downhill()
        (self.downhill_ratios, self.downhill_pds) = self.__get_downhill()
        self.current_base = 0.0
        self.current_base_ratio = np.array([0., 0., 0., 0., 0., 0.])
        self.duo = self.__get_duo()
        self.duo.find_new_pitches()
        # List of ratios to determine possible aggregate harmonic distance
        self.ratio_stack = self.__get_ratio_stack()

    def __repr__(self):
        return f"{self.current_base}\n{self.current_base_ratio}\n{self.duo.current_pitches}\n{self.duo.current_ratio()}"
    
    def __init_downhill(self):
        opt = DownhillOptimizer(n_points=1024)
        opt.populate_vectors([5, 5, 3, 2, 1, 1], (0.0, 1.0), hd_threshold=10.0)
        opt.populate_perms()
        opt.populate_distances()
        opt.populate_loss()
        return opt
    
    def __get_ratio_stack(self):
        return np.array([[0., 0., 0., 0., 0., 0.]])
    
    def __get_downhill(self):
        """
        Gets the "possible" downhill pitches, given a region of tolerance for
        every pitch class.

        Returns a tuple of the vectors followed by the pitch distances.
        """
        xs = np.linspace(0.0, 1.0, 1024)
        xys = np.stack([np.zeros_like(xs), xs], 1)
        logs = self.downhill_optimizer.optimize(xys)
        diffs = tf.abs(self.downhill_optimizer.pds[:, 1] - logs[:, None, 1])
        mins = tf.argmin(diffs, axis=1)
        uniques = tf.unique(mins).y
        vecs = tf.gather(self.downhill_optimizer.vectors, uniques)
        pds = hd.tenney.pd_aggregate_graph(vecs[None, :])
        return self.downhill_optimizer.session.run([vecs, pds])
    
    def __get_duo(self):
        with open("./init.yml", "r") as init:
            options = yaml.safe_load(init)
        duo = TromboneDuo()
        for trombone in duo.trombones:
            trombone.harmonic = options['starting_harmonic']
        duo.cache_vectors(options['prime_limits'], options['bounds'], options['hd_threshold'])
        duo.current_pitches = np.array([options['starting_pitches']])
        return duo
    
    def __get_current_base(self):
        pd = hd.tenney.pd_aggregate_graph(self.current_base_ratio[None, None, :])
        return self.__sess().run(pd)

    def __sess(self):
        return self.duo.optimizer.session
    
    def __get_downhill_hd_graph(self):
        ratios = np.append(self.ratio_stack, self.duo.current_ratio(), axis=0)
        diffs = ratios - self.downhill_ratios[:, None, :]
        return hd.tenney.hd_aggregate_graph(diffs)

    def __move_base_downhill(self):
        idx = self.__sess().run(tf.argmin(self.__get_downhill_hd_graph()))
        self.current_base_ratio += self.downhill_ratios[idx]
        self.current_base = self.__get_current_base()
        self.duo.current_pitches -= self.downhill_pds[0, idx]
        if self.current_base >= 1.0:
            self.current_base_ratio -= np.array([1., 0., 0., 0., 0., 0.])
            self.current_base = self.__get_current_base()
            self.duo.current_pitches += 1.0
    
    def __move_base_to_g(self):
        diff = G - self.current_base
        self.current_base_ratio = np.array([-1., 1., 0., 0., 0., 0.])
        self.current_base = self.__get_current_base()
        self.duo.current_pitches -= diff
    
    def __move_base_to_c(self):
        diff = 0.0 - self.current_base
        self.current_base_ratio = np.array([0., 0., 0., 0., 0., 0.])
        self.current_base = self.__get_current_base()
        self.duo.current_pitches -= diff
        # Re-initialize ratio stack
        self.ratio_stack = self.__get_ratio_stack()
    
    def step(self):
        if self.current_base == G:
            self.__move_base_to_c()
        elif self.current_base > F and self.current_base < A:
            self.__move_base_to_g()
        else:
            self.__move_base_downhill()
        self.duo.find_new_pitches(base=self.current_base)
    
    def decrement_harmonic(self, which=None):
        if which is None:
            for t in self.duo.trombones:
                current = t.harmonic
                ratio = np.log2(current / (current-1))
                t.harmonic = current - 1
        else:
            for idx in which:
                t = self.duo.trombones[idx]
                current = t.harmonic
        ratio = np.log2(current / (current-1))
            t.harmonic = current - 1
        self.duo.current_pitches -= ratio
        self.duo.find_new_pitches(base=self.current_base)
