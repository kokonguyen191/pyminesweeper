import itertools
import operator as op
import random
from functools import reduce

import numpy as np
from scipy.special import comb

from logic.engine.base import BaseEngine
from logic.game import Game


def ncr(n, r):
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer / denom


class SatSolverEngine(BaseEngine):

    def get_move(self, game_instance: Game):
        self.game = game_instance
        self.number_map = game_instance.get_map()
        self.number_map[game_instance.opened_map == 0] = -1
        self.original_number_map = self.number_map.copy()
        self.mine_map = np.zeros(self.number_map.shape)
        self.unfilled_map = 1 - game_instance.opened_map
        self.reduced = None
        self.total_unfilled = None
        self.nh = game_instance.rows
        self.nw = game_instance.cols

        check = True
        probabilities = dict()
        mine_set = set()
        safe_set = set()
        mine_dict = dict()
        safe_dict = dict()
        count = 0
        while check:
            temp_mine_set, temp_safe_set = self.solve_obvious_tiles()
            for _ in temp_mine_set:
                mine_set.add(_)
                self.mine_map[_[1], _[0]] = 1
                self.unfilled_map[_[1], _[0]] = 0
                mine_dict[_] = 1 + count * 2
            for _ in temp_safe_set:
                safe_set.add(_)
                self.unfilled_map[_[1], _[0]] = 0
                safe_dict[_] = 1 + count * 2
            temp_mine_set2, temp_safe_set2, blocks = self.solve_adjacents()
            temp_mine_set3, temp_safe_set3 = self.solve_blocks(blocks)
            for _ in temp_mine_set2.union(temp_mine_set3):
                mine_set.add(_)
                self.mine_map[_[1], _[0]] = 1
                self.unfilled_map[_[1], _[0]] = 0
                mine_dict[_] = 2 + count * 2
            for _ in temp_safe_set2.union(temp_safe_set3):
                safe_set.add(_)
                self.unfilled_map[_[1], _[0]] = 0
                safe_dict[_] = 2 + count * 2
            check = len(temp_mine_set) != 0 or len(temp_safe_set) != 0 \
                    or len(temp_mine_set2) != 0 or len(temp_safe_set2) != 0 \
                    or len(temp_mine_set3) != 0 or len(temp_safe_set3) != 0

            if not check and len(safe_set) == 0:
                temp_mine_set4, temp_safe_set4, probabilities = self.advanced_solve(blocks)
                for _ in temp_mine_set4:
                    mine_set.add(_)
                    self.mine_map[_[1], _[0]] = 1
                    self.unfilled_map[_[1], _[0]] = 0
                    mine_dict[_] = 10
                for _ in temp_safe_set4:
                    safe_set.add(_)
                    self.unfilled_map[_[1], _[0]] = 0
                    safe_dict[_] = 10
                if len(temp_safe_set4) > 0 or len(temp_mine_set4) > 0:
                    check = True

            count += 1

        out = None
        if len(safe_dict) == 0:
            # Make guess
            if len(probabilities) > 0:
                min_p = 1
                for p_list in probabilities:
                    for cell, p in p_list.items():
                        if min_p > p:
                            min_p = p
                            out = [cell]
            if out is None:
                out = random.sample(list(zip(*np.where(game_instance.opened_map == 0))), 1)
                out[0] = (out[0][1], out[0][0])
        else:
            out = list(safe_dict)

        for idx, _ in enumerate(out):
            out[idx] = _[1], _[0]
        return out

    def reduce_map(self):
        total_mines = np.zeros((self.nh, self.nw))
        padded_map = np.zeros((self.nh + 2, self.nw + 2))
        if self.mine_map is None:
            self.generate_maps()
        padded_map[1:self.nh + 1, 1:self.nw + 1] = self.mine_map
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                total_mines += padded_map[1 + i:1 + i + self.nh, 1 + j:1 + j + self.nw]
        self.reduced = self.number_map.copy()
        idx = self.number_map >= 0
        self.reduced[idx] = self.reduced[idx] - total_mines[idx]

    def get_total_unfilled(self):
        result = np.zeros((self.nh, self.nw))
        padded_map = np.zeros((self.nh + 2, self.nw + 2))
        if self.unfilled_map is None:
            self.generate_maps()
        padded_map[1:self.nh + 1, 1:self.nw + 1] = self.unfilled_map
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                result += padded_map[1 + i:1 + i + self.nh, 1 + j:1 + j + self.nw]
        self.total_unfilled = result

    def get_adjacent_cells(self, m, n, filled=False, get_all=False):
        if not filled:
            check_value = 1
        else:
            check_value = 0
        result = list()
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                mm = m + i
                nn = n + j
                if mm < 0 or mm >= self.nh or nn < 0 or nn >= self.nw:
                    continue
                if self.unfilled_map[mm, nn] == check_value or get_all:
                    result.append((nn, mm))
        return result

    def solve_obvious_tiles(self):
        self.reduce_map()
        self.get_total_unfilled()

        mine_set = set()
        safe_set = set()

        for i in range(self.nh):
            for j in range(self.nw):
                if self.reduced[i, j] > 0 and self.reduced[i, j] == self.total_unfilled[i, j]:
                    for _ in self.get_adjacent_cells(i, j):
                        mine_set.add(_)
                    continue
                if self.reduced[i, j] == 0 and self.total_unfilled[i, j] > 0:
                    for _ in self.get_adjacent_cells(i, j):
                        safe_set.add(_)
                    continue
        return mine_set, safe_set

    def solve_adjacents(self):
        self.reduce_map()
        self.get_total_unfilled()
        xs = np.multiply(self.unfilled_map, np.tile(np.arange(self.nw), (self.nh, 1)))[self.unfilled_map == 1].astype(
            int)
        ys = np.multiply(self.unfilled_map, np.tile(np.arange(self.nh), (self.nw, 1)).transpose())[
            self.unfilled_map == 1].astype(int)
        filled_adjacent_list = [self.get_adjacent_cells(y, x, filled=True) for x, y in zip(xs, ys)]
        i_cells = set()
        for _ in filled_adjacent_list:
            for __ in _:
                if self.mine_map[__[1], __[0]] == 1 or self.number_map[__[1], __[0]] == -1:
                    continue
                i_cells.add(__)
        i_cells = list(i_cells)
        i_unfilled_adjs = {(x, y): set(self.get_adjacent_cells(y, x, filled=False)) for x, y in i_cells}
        i_filled_adjs = dict()
        for x, y in i_cells:
            unfilled_adjacent_cells = i_unfilled_adjs[(x, y)]
            temp = set()
            for xx, yy in unfilled_adjacent_cells:
                temp = temp.union(set(self.get_adjacent_cells(yy, xx, filled=True)).intersection(set(i_cells)))
            i_filled_adjs[(x, y)] = temp

        mine_set = set()
        safe_set = set()
        blocks = list()

        for left, rights in i_filled_adjs.items():
            l_unfilled = i_unfilled_adjs[left]
            # Fill blocks
            if len(l_unfilled) > self.reduced[left[1], left[0]]:
                blocks.append((l_unfilled, self.reduced[left[1], left[0]]))

            for right in rights:
                r_unfilled = i_unfilled_adjs[right]
                l_diff = l_unfilled.difference(r_unfilled)
                r_diff = r_unfilled.difference(l_unfilled)

                # 1n
                if self.reduced[left[1], left[0]] == 1 \
                        and len(r_diff) == self.reduced[right[1], right[0]] - 1:
                    for _ in r_diff:
                        mine_set.add(_)
                    for _ in l_diff:
                        safe_set.add(_)

                if len(l_diff) != 0:
                    continue
                diff = self.reduced[right[1], right[0]] - self.reduced[left[1], left[0]]
                if diff == 0:
                    for _ in r_diff:
                        safe_set.add(_)
                elif len(r_diff) == diff:
                    for _ in r_diff:
                        mine_set.add(_)
                else:
                    blocks.append((r_diff, diff))
        return mine_set, safe_set, blocks

    def solve_blocks(self, blocks):
        mine_set = set()
        safe_set = set()
        checked_set = [_ for _, __ in blocks]
        checked_pair = list()
        while True:
            clone = blocks.copy()
            n = len(checked_set)
            for left, left_mines in clone:
                for right, right_mines in clone:
                    if left == right or not right.issubset(left):
                        continue
                    if (left, right) in checked_pair:
                        continue
                    checked_pair.append((left, right))
                    diff_set = left.difference(right)
                    diff = left_mines - right_mines
                    if diff == 0 and len(diff_set) > 0:
                        safe_set = safe_set.union(diff_set)
                    elif len(diff_set) == diff:
                        mine_set = mine_set.union(diff_set)
                    else:
                        if diff_set not in checked_set:
                            checked_set.append(diff_set)
                            blocks.append((diff_set, diff))
                if left not in checked_set:
                    checked_set.append(left)
            if len(checked_set) == n:
                break
        return mine_set, safe_set

    def advanced_solve(self, blocks):
        blocks = list(blocks)
        big_blocks = [_[0] for _ in blocks]
        blocks_list = [[_] for _ in blocks]
        while True:
            temp = list()
            temp_block_list = list()
            for idx0, cells in enumerate(big_blocks):
                flag = False
                for idx, big_block in enumerate(temp):
                    if len(big_block.intersection(cells)) > 0:
                        temp[idx] = temp[idx].union(cells)
                        temp_block_list[idx].extend(blocks_list[idx0])
                        flag = True
                        break
                if not flag:
                    temp.append(cells)
                    temp_block_list.append(blocks_list[idx0])
            blocks_list = temp_block_list
            if len(big_blocks) == len(temp):
                break
            big_blocks = temp
        total_mine_set = set()
        total_safe_set = set()
        all_probabilities = list()
        for big_block, sub_blocks in zip(big_blocks, blocks_list):
            mine_set, safe_set, probabilities = self.evaluate_all_possibilities(big_block, sub_blocks)
            total_mine_set = total_mine_set.union(mine_set)
            total_safe_set = total_safe_set.union(safe_set)
            all_probabilities.append(probabilities)
        return total_mine_set, total_safe_set, all_probabilities

    def evaluate_all_possibilities(self, big_block, blocks):
        TESTS = 1000

        all_cells = list()
        for block, _ in blocks:
            for cell in block:
                if cell not in all_cells:
                    all_cells.append(cell)
        idx_dict = {cell: idx for idx, cell in enumerate(all_cells)}
        reverse_idx_dict = {idx: cell for idx, cell in enumerate(all_cells)}
        blocks = [([idx_dict[cell] for cell in block], int(mines)) for block, mines in blocks]
        big_block = {idx_dict[cell] for cell in big_block}

        min_possibilities = float('inf')
        min_partitions = list()
        min_loose_cells = set()
        for _ in range(TESTS):
            permutation = np.random.permutation(blocks)
            partitions = list()
            clone = set(big_block)
            for cells, mines in permutation:
                if set(cells).issubset(clone):
                    clone = clone.difference(cells)
                    partitions.append((cells, mines))
            loose_cells = clone
            n_possibilities = 2 ** len(loose_cells)
            for group, mine_count in partitions:
                n_possibilities *= ncr(len(group), int(mine_count))
            if n_possibilities < min_possibilities:
                min_possibilities = n_possibilities
                min_partitions = partitions
                min_loose_cells = loose_cells

        if min_loose_cells is None or min_partitions is None:
            return set(), set(), dict()

        if min_possibilities > 1500000:
            return set(), set(), dict()

        if len(min_partitions) > 0:
            choices = [itertools.combinations(*partition) for partition in min_partitions]
        else:
            choices = list()
        temp = list()
        for i in range(len(min_loose_cells) + 1):
            temp.extend(itertools.combinations(min_loose_cells, i))
        choices.append(temp)
        choices = itertools.product(*choices)
        choices = list(choices)
        if min_possibilities > 200000:
            min_possibilities = 200000
            choices = random.sample(choices, min_possibilities)

        for idx, choice in enumerate(choices):
            temp = list()
            for _ in choice:
                temp.extend(_)
            choices[idx] = temp
        simulation_array = np.zeros((int(min_possibilities), len(all_cells) + len(blocks) + 1))
        for idx, choice in enumerate(choices):
            simulation_array[idx, choice] = 1
        for idx, (block, mines) in enumerate(blocks):
            simulation_array[:, len(all_cells) + idx] = simulation_array[:, block].sum(axis=1) == mines
        simulation_array[:, -1] = np.all(simulation_array[:, len(all_cells):len(all_cells) + len(blocks)], axis=1)
        possible_outcomes = simulation_array[simulation_array[:, -1] == 1, :len(all_cells)]
        mine_sum = possible_outcomes.sum(axis=1)
        unique_mine_sum, unique_mine_sum_count = np.unique(mine_sum, return_counts=True)
        unfilled_cell = self.unfilled_map.sum() - len(big_block)
        mine_left = self.game.n_mines - self.game.mine_map.sum()
        if mine_left <= 0:
            unique_mine_sum_max = unique_mine_sum.max()
            mine_left = int((unfilled_cell - unique_mine_sum_max) * 0.25 + unique_mine_sum_max)
        comb_list = [comb(unfilled_cell, mine_left - _) for _ in unique_mine_sum]
        comb_dict = {_: __ for _, __ in zip(unique_mine_sum, comb_list)}
        total_possibilities = 0
        for out_comb, in_comb in zip(comb_list, unique_mine_sum_count):
            total_possibilities += out_comb * in_comb
        possible_outcomes *= np.array([comb_dict[_] for _ in mine_sum])[:, None]
        possible_outcomes /= total_possibilities

        predictions = np.sum(possible_outcomes, axis=0)
        mine_indices = np.arange(len(all_cells))[predictions == 1]
        safe_indices = np.arange(len(all_cells))[predictions == 0]

        if min_possibilities != 200000:
            mine_set = {reverse_idx_dict[idx] for idx in mine_indices}
            safe_set = {reverse_idx_dict[idx] for idx in safe_indices}
            probabilities = {
                reverse_idx_dict[idx]: predictions[idx]
                for idx in np.arange(len(all_cells))[np.logical_and(predictions > 0, predictions < 1)]
            }
        else:
            mine_set = set()
            safe_set = set()
            probabilities = {
                reverse_idx_dict[idx]: predictions[idx]
                for idx in range(len(all_cells))
            }
        return mine_set, safe_set, probabilities
