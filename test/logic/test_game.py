from unittest import TestCase

import numpy as np

from exceptions import InvalidGame
from logic.game import Game


class TestGame(TestCase):
    def test_initialize_map(self):
        def test_no_init_cell():
            for i in range(1, 16):
                game = Game(3, 6, i)
                self.assertEqual(game.mine_map.sum(), i)

        def test_init_cell():
            for i in range(3):
                for j in range(6):
                    for _ in range(1, 18):
                        game = Game(3, 6, _, (i, j))
                        self.assertEqual(game.mine_map[i, j], 0)

        def test_exceptions():
            # Too few mines
            self.assertRaises(InvalidGame, lambda: Game(10, 10, 0))
            self.assertRaises(InvalidGame, lambda: Game(10, 10, -5))
            # Too many mines
            self.assertRaises(InvalidGame, lambda: Game(10, 10, 100))
            self.assertRaises(InvalidGame, lambda: Game(10, 10, 150))
            # Init mine outside range
            self.assertRaises(InvalidGame, lambda: Game(10, 10, 50, (-5, 5)))
            self.assertRaises(InvalidGame, lambda: Game(10, 10, 50, (5, -5)))
            self.assertRaises(InvalidGame, lambda: Game(10, 10, 50, (5, 15)))
            self.assertRaises(InvalidGame, lambda: Game(10, 10, 50, (15, 5)))

        test_no_init_cell()
        test_init_cell()
        test_exceptions()

    def test_bfs_number_cell(self):
        game = Game(5, 5, 1)
        game.mine_map = np.array([
            [1, 1, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ])
        game.fill_numbers()
        expected = {(3, 2), (3, 3), (3, 1), (2, 1), (2, 3), (4, 3), (2, 2), (4, 2), (4, 1)}
        actual1 = game.bfs((4, 2))
        actual2 = game.bfs((3, 2))
        self.assertEqual(actual1, actual2)
        self.assertEqual(actual1, expected)

    def test_open_cell_and_3bvs(self):
        game = Game(5, 5, 1)
        game.mine_map = np.array([
            [1, 1, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ])
        game.fill_numbers()
        game.open_cell((4, 2))
        self.assertEqual(game.get_3bvs(), 10)
