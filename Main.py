import gym
from Classes import Agent
import numpy as np

if __name__ == '__main__':
    env = gym.make('Taxi-v3')
    n_actions = env.action_space.n
    agent = Agent(gamma= 0.9, lr= 0.001, eps=0.1, n_states= env.observation_space.n, n_actions=n_actions)
    n_games = 1000000
    scores = []

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        timer = 0
        while not done:
            action = agent.choose_action(observation)
            timer += 1
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.learn(state = observation, action=action,next_state=observation_, reward = reward, done=done)
            observation = observation_
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        print("Episode: {}, Score: {}, Timer: {}, Average: {}".format(i, score, timer,avg_score))
