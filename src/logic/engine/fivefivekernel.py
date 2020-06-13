import numpy as np
from scipy import signal

from logic.engine.base import BaseEngine


class FiveFiveKernelEngine(BaseEngine):

    def __init__(self, kernel):
        self.kernel = kernel

    def get_move(self, game_instance):
        convolved = signal.convolve2d(game_instance.get_map(), self.kernel, mode='same')
        average = signal.convolve2d(game_instance.get_map(), np.ones((5, 5)), mode='same')
        convolved[average == 0] = 1
        least_risk_masked_idx = convolved[game_instance.opened_map == 0].argmin()
        next_move_cell = list(zip(*np.where(game_instance.opened_map == 0)))[least_risk_masked_idx]
        return next_move_cell
