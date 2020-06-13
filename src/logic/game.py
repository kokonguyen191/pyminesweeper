from enum import Enum

import numpy as np
from scipy import signal

from exceptions import InvalidGame


class CellState(Enum):
    UNOPENED = 0
    NUMBER = 1
    FLAG = 2


class Game:
    """
    Game object
    Contains minimum data to run a game, 0-based index.
    For 1D representation, concatenate the rows, iterate by columns. E.g. [[0,1,2],[3,4,5],[6,7,8]]
    """

    def __init__(self, rows, cols, n_mines, init_cell=None):
        """
        Ctor
        :type init_cell: (row, col) tuple
        """
        if not (0 < n_mines < rows * cols):
            raise InvalidGame("Invalid value of mines {} for size {}x{}".format(n_mines, rows, cols))

        self.rows = rows
        self.cols = cols
        self.n_mines = n_mines
        self.mine_map = np.zeros((rows, cols))

        self.opened_map = np.zeros((rows, cols))
        self.number_map = np.zeros((rows, cols))

        self.is_done = False
        self.is_lost = False
        self.is_won = False

        if init_cell is not None:
            self.initialize(init_cell)

    def initialize(self, init_cell):
        if init_cell is not None:
            if not isinstance(init_cell, tuple) \
                    or len(init_cell) != 2 \
                    or not (0 <= init_cell[0] < self.rows) \
                    or not (0 <= init_cell[1] < self.cols):
                raise InvalidGame("Invalid init cell ", init_cell)
        self.initialize_map(init_cell)
        self.fill_numbers()
        if init_cell is not None:
            self.open_cell(init_cell)

    def initialize_map(self, init_cell=None):
        n_sample = self.cols * self.rows if init_cell is None else self.cols * self.rows - 1
        sampled_indices = np.random.choice(n_sample, self.n_mines, replace=False).tolist()
        if init_cell is not None:
            sampled_indices = [(_ + 1 + init_cell[0] * self.cols + init_cell[1]) % (self.cols * self.rows) for _ in
                               sampled_indices]
        for sampled_index in sampled_indices:
            sampled_row, sampled_col = np.divmod(sampled_index, self.cols)
            self.mine_map[sampled_row, sampled_col] = 1

    def fill_numbers(self):
        self.number_map = signal.convolve2d(self.mine_map,
                                            np.array([[1, 1, 1],
                                                      [1, 0, 1],
                                                      [1, 1, 1]]),
                                            mode='same')
        self.number_map[self.mine_map == 1] = -1

    def flag_cell(self, cell):
        row, col = cell
        self.opened_map[row, col] = CellState.FLAG.value

    def open_cell(self, cell):
        row, col = cell
        if self.mine_map[row, col] == 1:
            self.is_lost = True
            self.is_done = True
            return True
        if self.opened_map[row, col] == 1:
            return False
        opened_cells = self.bfs(cell)
        for opened_cell in opened_cells:
            row, col = opened_cell
            self.opened_map[row, col] = CellState.NUMBER.value
        return False

    def check_clear_condition(self):
        if self.is_done:
            return True
        if self.get_3bvs() == 0:
            self.is_won = True
            self.is_done = True
            return True
        return False

    def get_3bvs(self):
        unopened_mask = self.opened_map == CellState.UNOPENED.value
        non_mine_mask = self.mine_map == 0
        merged_mask = np.logical_and(unopened_mask, non_mine_mask)
        rows, cols = np.where(merged_mask)
        unopened_number_cells = set(zip(rows, cols))
        zeros_unopened_number_cells = {
            (row, col)
            for (row, col) in unopened_number_cells
            if self.number_map[row, col] == 0
        }

        threebv_count = 0
        cells_opened_with_zeros = set()
        while len(zeros_unopened_number_cells) > 0:
            cell = zeros_unopened_number_cells.pop()
            threebv_count += 1
            explored = self.bfs(cell)
            cells_opened_with_zeros = cells_opened_with_zeros.union(explored)
            zeros_unopened_number_cells = zeros_unopened_number_cells.difference(explored)
        unopened_number_cells = unopened_number_cells.difference(cells_opened_with_zeros)
        threebv_count += len(unopened_number_cells)
        return threebv_count

    def __bfs_internal(self, cell, visited):
        visited.add(cell)
        row, col = cell
        if self.number_map[row, col] == -1:
            return {}
        if self.number_map[row, col] > 0:
            return {cell}
        lr = row - 1 if row > 0 else row
        rr = row + 1 if row < self.rows - 1 else row
        lc = col - 1 if col > 0 else col
        rc = col + 1 if col < self.cols - 1 else col
        result_set = {cell}
        for r in range(lr, rr + 1):
            for c in range(lc, rc + 1):
                adj_cell = (r, c)
                if adj_cell in visited:
                    continue
                result_set = result_set.union(self.__bfs_internal(adj_cell, visited))
        return result_set

    def bfs(self, cell):
        return self.__bfs_internal(cell, set())

    def get_map(self):
        return self.opened_map * self.number_map

    def clone(self):
        game = Game(self.rows, self.cols, self.n_mines)
        game.mine_map = self.mine_map.copy()
        game.opened_map = self.opened_map.copy()
        game.number_map = self.number_map.copy()
        return game

    def __str__(self):
        print_map = self.opened_map * self.number_map
        print_map[self.opened_map == 0] = np.nan
        return str(print_map)

    def save_game(self, filename):
        np.save(filename, self.mine_map)


if __name__ == '__main__':
    game = Game(10, 10, 10, (4, 4))
    # np.savetxt("foo.csv", game.mine_map.astype(int).tolist(), delimiter=",")
    game.mine_map = np.genfromtxt("foo.csv", delimiter=',')
    game.fill_numbers()
    while not game.check_clear_condition():
        print(game)
        inp = input("Make your move: ")
        inp = inp.split()
        row = int(inp[0])
        col = int(inp[1])
        fail = game.open_cell((row, col))
        if fail:
            print("Boom, you are dead")
    if not fail:
        print("You won!")
