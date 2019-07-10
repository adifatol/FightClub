import retro
import numpy as np
import random
import cv2

class GenesisAgent():
    def __init__(self, env, bin_state=True, res_state=True):
        self.bin_state = bin_state
        self.res_state = res_state
        self.action_space = self._get_actions()
        self.action_size = len(self.action_space)
        self.state_size = self._get_state_size(env)
        
    def _get_actions(self):
        action_atk = [["B"], ["A"], ["C"], ["Y"], ["X"], ["Z"]]
        action_mov = [["UP"], ["DOWN"], ["LEFT"], ["RIGHT"]]
        # the player is also able to press button combinations
        action_atk_cmbo = [["DOWN"] + atk for atk in action_atk] # sweeping kicks/punches 
        action_mov_cmbo = [["UP", "LEFT"], ["UP", "RIGHT"]] # left/right jumps
        # **Note**: JUMP-ATTACKS are sequential actions not instant
        actions_raw = action_atk + action_mov + action_atk_cmbo + action_mov_cmbo
        # the env reuquires discrete values, no strings    
        return self._discretize_actions(actions_raw)

    def _discretize_actions(self, action_space):
        btns = ["B", "A", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        # convert string to number with respect to order in btns
        ack_position = [[btns.index(btn) for btn in ack] for ack in action_space]
        # create an empty array of possible actions
        ack_discrete = np.zeros((len(action_space),len(btns)), dtype=int)
        # each row is then populated based on ack_position
        for i, pos in enumerate(ack_position):
            np.put(ack_discrete[i], pos, [1]*len(pos))
        return ack_discrete
    
    def _get_state_size(self, env):
        obs = env.reset()    
        rnd_move = self.pick_rnd_move()
        state, reward, done, info = env.step(self.action_space[rnd_move])
        compact_state = self.reduce_dim(state)
        return compact_state.shape

    def reduce_dim(self, state):
        if self.bin_state: state = self.binarize_state(state)
        if self.res_state: state = self.resize_state(state)
        return state
    
    def binarize_state(self, state):
        state = cv2.cvtColor(cv2.resize(state, state.shape[1::-1]), cv2.COLOR_BGR2GRAY)
        ret,state_bin = cv2.threshold(state,70,140,cv2.THRESH_BINARY)
        state_bin = state_bin[50:-15,:] # magic numbers for dim reduction
        return state_bin
    
    def resize_state(self, state):
        mini = min(state.shape)
        return cv2.resize(state, (mini,mini))

    def pick_rnd_move(self):
        return random.choice(range(len(self.action_space)))
