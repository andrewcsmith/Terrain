from trombone_duo import *
import yaml

duo = TromboneDuo()
with open("./init.yml", "r") as init:
    options = yaml.safe_load(init)

duo.cache_vectors(options['prime_limits'], options['bounds'])
duo.current_pitches = np.array([options['starting_pitches']])
