import random
import numpy as np
from collections import deque
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Concatenate, Flatten

from genesis_agent import GenesisAgent

class DQNAgent(GenesisAgent):
    def __init__(self, env):
        super().__init__(env)
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.eps = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        shape_x = self.state_size[0]
        shape_y = self.state_size[1]
        input_img = Input(shape=(shape_x, shape_y, 1))
        ### 1st layer
        layer_1 = Conv2D(10, (1,1), padding='same', activation='relu')(input_img)
        layer_1 = Conv2D(10, (3,3), padding='same', activation='relu')(layer_1)

        layer_2 = Conv2D(10, (1,1), padding='same', activation='relu')(input_img)
        layer_2 = Conv2D(10, (5,5), padding='same', activation='relu')(layer_2)

        layer_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_img)
        layer_3 = Conv2D(10, (1,1), padding='same', activation='relu')(layer_3)

        mid_1 = Concatenate(axis = 3)([layer_1,layer_2, layer_3])
        
        flat_1 = Flatten()(mid_1)

        dense_1 = Dense(1200, activation='relu')(flat_1)
        dense_2 = Dense(600, activation='relu')(dense_1)
        dense_3 = Dense(150, activation='relu')(dense_2)
        output = Dense(self.action_size, activation='softmax')(dense_3)
        
        model = Model([input_img], output)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model
        
    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def format_state(self, state):
        compact_state = super().reduce_dim(state)
        compact_3d_state = np.expand_dims(compact_state,axis=2)
        return np.array([compact_3d_state.tolist()])
        
    def remember(self, state, action, reward, next_state, done):
        state = self.format_state(state)
        next_state = self.format_state(next_state)
        self.memory.append((state, action, reward, next_state, done))
    
    def get_move(self, state):
        if np.random.rand() <= self.eps:
            return super().pick_rnd_move()
        state = self.format_state(state)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def move_to_act(self, move):
        return self.action_space[move]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                # a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
