import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class KinematicCar(gym.Env):
    def __init__(self):
        self.L = 1.0
        self.dt = 0.1
        self.max_y = 5.0
        self.min_y = -5.0
        self.max_speed = 30.0
        self.min_speed = 0.0
        self.max_theta = np.pi/2
        self.min_theta = -np.pi/2

        self.max_steer = np.pi/2
        self.min_steer = -np.pi/2
        self.max_throttle = 3.0
        self.min_throttle = -10.0

        self.theta_constraint_penalty = -10000
        self.y_constraint_penalty = -10000

        self.weight_y = 10.0
        self.weight_v = 2.0
        self.weight_heading = 2.0

        self.weight_throttle = 1.0
        self.weight_steer = 3.0
        self.v_ref = 10.0

        self.d_steer_weight = 5.0
        self.d_throttle_weight = 5.0

        self.low_obs = np.array([self.min_y, self.min_theta, self.min_speed, 0.0])
        self.high_obs = np.array([self.max_y, self.max_theta, self.max_speed, self.v_ref])

        self.low_act = np.array([self.min_steer, self.min_throttle])
        self.high_act = np.array([self.max_steer, self.max_throttle])

        self.viewer = None

        self.action_space = spaces.Box(self.low_act, self.high_act)
        self.observation_space = spaces.Box(self.low_obs, self.high_obs)

        self.heading_ref = 0.0
        self.y_ref = 0.0

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # TODO: Need some kind of trajectory generator for getting up to speed
    def step(self, action):
        reward = 0
        done = False
        steer = action[0]
        throttle = action[1]
        
        x_ = self.x + self.state[2] * np.cos(self.state[1]) * self.dt
        y_ = self.state[0] + self.state[2] * np.sin(self.state[1]) * self.dt
        theta_ = self.state[1] + self.state[2] /self.L * steer * self.dt

        speed_ = max(min(self.state[2] + throttle * self.dt, self.max_speed), self.min_speed)

        self.state[0] = y_
        self.x = x_
        self.state[1] = theta_
        self.state[2] = speed_
        self.t += self.dt
        
        e_y = abs(self.y_ref - self.state[0])
        this_v_ref = 10.0#self.v_t_ref(self.t)
        e_v = abs(this_v_ref - self.state[2])
        self.state[3] = this_v_ref
        e_theta = abs(self.heading_ref - self.state[1])
        d_steer = abs(self.last_steer - action[0])
        d_throttle = abs(self.last_throttle- action[1])
        self.last_throttle = throttle
        self.last_steer = steer
        # Constraints
        if theta_ > self.max_theta or theta_ < self.min_theta:
            reward += self.theta_constraint_penalty
            done = True

        if y_ > self.max_y or y_ < self.min_y:
            reward += self.y_constraint_penalty
            done = True

        state_cost = self.weight_y*e_y**2 + self.weight_v*e_v**2 + self.weight_heading*e_theta**2
        input_cost = self.weight_throttle*throttle**2 + self.weight_steer*steer**2 + self.d_throttle_weight*d_throttle**2 + self.d_steer_weight*d_steer**2

        reward += -state_cost -input_cost


        return self.state,reward, done, {}

    def v_t_ref(self, t, v_0 = 0.0):
        return self.v_ref*(1 - np.exp(-t/100))

    def reset(self):
        self.x = 0.0
        self.t = 0.0
        self.last_steer = 0.0
        self.last_throttle = 0.0
        self.state = np.array([self.np_random.uniform(low=-0.1, high=0.1), 0, 10.0, 0])
        return np.array(self.state)

    def render(self, mode='human'):
        pass

    def close(self):
        pass