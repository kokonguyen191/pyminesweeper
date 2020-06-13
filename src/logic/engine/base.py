from abc import ABC, abstractmethod


class BaseEngine(ABC):
    @abstractmethod
    def get_move(self, game_instance):
        return
