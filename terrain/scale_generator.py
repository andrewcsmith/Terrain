from structure import *
from optimizer import *
from matplotlib import pyplot as plt
import harmonic_distance as hd
from ratio_presenter import *
import yaml
import json

class ScaleGenerator():
    def __init__(self, n_points=1, dimensions=4):
        self.n_points = n_points
        self.dimensions = dimensions
        self.opt = LastDimensionOptimizer(n_points=self.n_points,dimensions=self.dimensions)
        with open("output.json", "r") as f:
            self.harmony_data = json.load(f)
        self.base_ratios = np.array(self.harmony_data['base_ratios'])
        self.duo_ratios = np.array(self.harmony_data['duo_ratios'])
        self.chords = np.concatenate([self.base_ratios[:, None, :], self.duo_ratios], 1)
        self.limits = [16, 12, 2, 2, 1, 0]
        self.opt.populate_vectors(self.limits, (0.0, 7.0))
        # Vectors is the list of possible pitches that could be harmonized with.
        self.vectors = self.opt.session.run(self.opt.vectors)

    def generate(self, n=0, c=1.0e-2, r=(0.0, 7.0)):
        """
        n: the number of the chord to use to generate
        """
        tiled = np.tile(self.chords[n], [self.vectors.shape[0], 1, 1])
        perms = np.concatenate([tiled, self.vectors[:, None, :]], axis=1)
        self.opt.load_perms(perms)
        self.opt.populate_distances()
        self.opt.c = c
        self.opt.populate_loss()
        xs = np.linspace(r[0], r[1], self.n_points)
        log_tiled = np.tile(np.log2((hd.PRIMES[:self.chords.shape[-1]] ** self.chords[n]).prod(1)), [self.n_points, 1])
        log_pitches = np.concatenate([log_tiled, xs[:, None]], 1)
        self.opt.assign_starting_pitches(log_pitches)
        ys = self.opt.session.run(self.opt.loss)
        return (xs, ys)
