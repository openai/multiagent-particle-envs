import numpy as np
from pyglet.window import key

# individual agent policy


class Policy(object):
    def __init__(self):
        pass

    def action(self, obs):
        raise NotImplementedError()