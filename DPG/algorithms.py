import numpy as np

class COPDAQ:
    def __init__(self, action_dim, state_dim):
        self.theta = np.zeros((state_dim,1))
        self.w = np.zeros((state_dim,1))
        self.v = np.zeros((state_dim,1))
        self.gamma = 0.09
        self.alpha = 0.1

    def step(self, s_features, next_s_features, reward, action):
        det_action = np.dot(self.theta.T, s_features)
        current_Q = (action- det_action) * np.dot(s_features.T, self.w) + np.dot(s_features.T, self.v)
        next_Q = np.dot(next_s_features.T, self.v)
        # TD error
        delta = reward + self.gamma * next_Q - current_Q
        # theta gradient update
        self.theta +=  self.alpha * s_features * np.dot(s_features.T,self.w)
        # w gradient update
        self.w += self.alpha * delta * action * s_features
        # v gradient update
        self.v += self.alpha * delta * s_features

    def mean(self, s_features):
        return np.dot(self.theta.T, s_features)