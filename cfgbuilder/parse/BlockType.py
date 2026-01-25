from enum import Enum


class BlockType(Enum):
    UNDEFINED = 1
    COMMON = 2
    DISPATCHER = 3
    FALLBACK = 4
    START = 5
    LOADER = 6
    STOP = 7

    MONEY_IN = 8
    STATE_CHANGE = 9
    SUICIDE_BLOCK = 10
    TERMINAL_FAILURE = 11

    def __str__(self) -> str:
        return self.name
