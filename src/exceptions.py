class InvalidGame(Exception):
    def __init__(self, msg, *args):
        super().__init__(msg, args)
