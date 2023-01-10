import numpy as np


class BasicLandmark():
    def __init__(self, id="agent", type="obstacle", initial_conditions=None):
        
        self.id = id
        self.type = type
        # Data used to reset agent
        self._init_pos = initial_conditions['position']\
            if initial_conditions is not None and 'position' in initial_conditions\
                else np.zeros(2)  # [x, y]
        self._init_color = initial_conditions['color']\
            if initial_conditions is not None and 'color' in initial_conditions\
                else np.zeros(4)  # [r, g , b, alpha]
        self._init_size = initial_conditions['size']\
            if initial_conditions is not None  and 'size' in initial_conditions\
                else 0.05


    def reset(self):
        self.position = self._init_pos
        self.color = self._init_color
        self.size = self._init_size
