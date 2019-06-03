import numpy as np
import cv2

class Agent():
    def __init__(self, env):
        self.btns = ["B", "A", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        self.actions = self.gen_streetfighter_actions()
        self.action_size = len(self.actions)
        self.state_size = self.gen_streetfighter_states(env)

    def gen_streetfighter_actions(self):
        action_atk = [["B"], ["A"], ["C"], ["Y"], ["X"], ["Z"]]
        action_mov = [["UP"], ["DOWN"], ["LEFT"], ["RIGHT"]]

        action_atk_cmbo = [["DOWN"] + atk for atk in action_atk] # JUMP-ATTACKS are sequential actions not instant
        action_mov_cmbo = [["UP", "LEFT"], ["UP", "RIGHT"], ["DOWN", "LEFT"], ["DOWN", "RIGHT"]]

        actions_raw = action_atk + action_mov + action_atk_cmbo + action_mov_cmbo

        return self.discretize_actions(actions_raw)

    def gen_streetfighter_states(self, env):
        obs = env.reset()
        state, reward, done, info = env.step([0]*12)
        return self.binarize_state(state).shape

    def discretize_actions(self, actions):
        ack_position = [[self.btns.index(btn) for btn in ack] for ack in actions]
        ack_discrete = np.zeros((len(actions),len(self.btns)), dtype=int)

        for i, pos in enumerate(ack_position):
            np.put(ack_discrete[i], pos, [1]*len(pos))

        return ack_discrete

    def binarize_state(self, state):
        state = cv2.cvtColor(cv2.resize(state, state.shape[1::-1]), cv2.COLOR_BGR2GRAY)
        ret,state_bin = cv2.threshold(state,70,140,cv2.THRESH_BINARY)
        state_bin = state_bin[50:-15,:] # magic numbers for dim reduction
        return state_bin

    def resize_state(self, state):
        mini = min(state.shape)
        return cv2.resize(state, (mini,mini))
