from enum import Enum


class Direction(str, Enum):
    CE = "CE"
    PE = "PE"
    WAIT = "WAIT"


class Environment(str, Enum):
    BACKTEST = "BACKTEST"
    RESEARCH = "RESEARCH"
    UAT = "UAT"
    PAPER = "PAPER"
    LIVE = "LIVE"
