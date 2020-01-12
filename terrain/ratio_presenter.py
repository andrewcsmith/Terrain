import yaml
from functools import reduce
import numpy as np
import harmonic_distance as hd

with open("./ratio_presenter.yml", "r") as init:
    RATIO_PRESENTER_OPTIONS = yaml.safe_load(init)

class RatioPresenter:
    def __init__(self, vector):
        n_primes = vector.shape[-1]
        self.basenames = RATIO_PRESENTER_OPTIONS['basenames']
        self.primesteps = np.array(RATIO_PRESENTER_OPTIONS['primesteps'])[:n_primes]
        self.primedegrees = np.array(RATIO_PRESENTER_OPTIONS['primedegrees'])[:n_primes]
        self.primes = hd.PRIMES[:n_primes]
        self.primealterations = RATIO_PRESENTER_OPTIONS['primealterations']
        self.vector = vector
        self.factors = dict(zip(self.primes, self.vector))
    
    def __repr__(self):
        return self.get_basename() + self.get_alteration() + self.get_octave()

    def get_steps(self):
        return np.sum(self.primesteps * self.vector)
    
    def get_basename(self):
        return self.basenames[int(self.get_steps())]

    def get_alteration(self):   
        alteration = ""
        for p, e in self.factors.items():
            if p > 3:
                if e > 0:
                    if p == 11:
                        alteration += self.__collect_string(int(e), "U" + self.primealterations[p])
                    else:
                        alteration += self.__collect_string(int(e), "D" + self.primealterations[p])
                elif e < 0:
                    if p == 11:
                        alteration += self.__collect_string(int(e) * -1, "D" + self.primealterations[p])
                    else:
                        alteration += self.__collect_string(int(e) * -1, "U" + self.primealterations[p])

        return alteration

    def get_octave(self):
        degrees = np.sum(self.primedegrees * self.vector)
        octave = ""
        if degrees > 0:
            octave += reduce(lambda l, r: l + "'", range(int(np.floor(degrees / 7.0))), "")
        elif degrees < 0:
            octave += reduce(lambda l, r: l + ",", range(int(np.floor(degrees / -7.0))), "")
        return octave
    
    @staticmethod
    def __collect_string(n_times, input):
        return reduce(lambda l, r: l + input, range(n_times), "")
