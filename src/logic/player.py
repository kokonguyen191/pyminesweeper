import logging

from logic.engine.base import BaseEngine
from logic.game import Game


class Player:
    engine: BaseEngine
    game: Game

    def __init__(self, game, engine):
        self.game = game
        self.engine = engine

    def make_move(self):
        next_move_cell = self.engine.get_move(self.game)
        fail = False
        if not isinstance(next_move_cell, list):
            next_move_cell = [next_move_cell]
        for _ in next_move_cell:
            logging.debug("Making move {}".format(_))
            res = self.game.open_cell(_)
            fail = fail and res
        return fail

    def play(self):
        original_3bvs = self.game.get_3bvs()
        while not self.game.check_clear_condition():
            result = self.make_move()
        return 1 - self.game.get_3bvs() / original_3bvs
