from datetime import datetime

import numpy as np

from logic.engine.satsolver import SatSolverEngine
from logic.game import Game
from logic.player import Player

# logging.basicConfig(level=logging.DEBUG)
np.set_printoptions(linewidth=160)

count = 0
with open("foo.csv", 'w') as fw:
    while True:
        count += 1
        game = Game(24, 30, 217, (0, 0))
        solver = SatSolverEngine()
        player = Player(game, solver)
        start = datetime.now()
        play = player.play()
        if play == 1:
            print("Done in {} tries".format(count))
            count = 0
        elapsed = (datetime.now() - start).total_seconds()
        print("Clear {}% in {}".format(100 * play, elapsed))
        fw.write("{},{}\n".format(play, elapsed))
