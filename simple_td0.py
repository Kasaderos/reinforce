import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EPISODE_LEN = 60 

# exponential moving average
def ema(data, window = 14):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()

    ema = np.convolve(data, weights, mode='full')[:len(data)]
    ema[:window] = ema[window]

    return ema

class Frame:
    
    def __init__(self, y, v1, v2):
        self.y = y
        self.v1 = v1
        self.v2 = v2

class Episode:

    def __init__(self, y, I1, I2):
        self.y = y
        self.I1 = I1
        self.I2 = I2

    def get_frame(self, i):
        return Frame(self.y[i], self.I1[i], self.I2[i])

# y - price
# v1 - value of the first indicator
# v2 - value of the second
# get_state returns one of the state number 
def get_state(y, v1, v2):
    if v1 > v2 and v1 > y and v2 > y:
        return 0
    elif v1 > v2 and v1 > y and v2 < y:
        return 1
    elif v1 < v2 and v1 > y and v2 > y:
        return 2
    elif v1 < v2 and v1 > y and v2 < y:
        return 3
    elif y > v1 and y > v2 and v1 > v2: 
        return 4
    elif y > v1 and y > v2 and v1 < v2: 
        return 5
    else:
        return 6
    
def generator_episodes(y):
    while True:
        i = np.random.randint(0, len(y)-EPISODE_LEN) 

        yi = y[i:i+EPISODE_LEN]
        ema14i = ema(yi, 14)
        ema28i = ema(yi, 28)

        yield Episode(yi, ema14i, ema28i) 

def get_reward(frame, next_frame, cache=100):
    return (next_frame.y/frame.y - 1) * cache

# get_v returns value of state 's'
def get_v(s):
    if s not in V:
        V[s] = 0.0
        return 0.0

    return V[s]

class SampleEpisode:
    
    def __init__(self, max_episodes = 5):
        self.episodes = []
        self.ptr = 0
        self.max_episodes = max_episodes

    def add_episode(self, episode):
        if len(self.episodes) >= self.max_episodes:
            return
        self.episodes.append(episode)

    def plot(self):
        self.ptr += 1 
        if self.ptr >= len(self.episodes):
            self.ptr = 0
        e = self.episodes[self.ptr]
        x = np.arange(EPISODE_LEN)
        fig, ax = plt.subplots()

        y = e.y 
        v1 = e.I1 
        v2 = e.I2 
        for i in range(EPISODE_LEN-1):
            if y[i] > v1[i] and y[i] > v2[i] and v1[i] > v2[i]: 
                ax.plot([x[i], x[i+1]], [y[i], y[i+1]], color='green')
            else:
                ax.plot([x[i], x[i+1]], [y[i], y[i+1]], color='red')
                
        ax.plot(x, v1, 'k--', label='I1')
        ax.plot(x, v2, 'b--', label='I2')
        ax.legend(loc='best')
        plt.show()

# train RL TD(0) algorithm
def train():
    M = 1000
    alpha = 0.5
    gamma = 1.0
    lag = 14 # weekdays
    for i in range(M):
        episode = next(gen_episode)
        if M > 500:
            sample.add_episode(episode)

        t = 0
        for i in range(lag, EPISODE_LEN):
            frame = episode.get_frame(t)
            s = get_state(frame.y, frame.v1, frame.v2)

            next_frame = episode.get_frame(i)
            next_s = get_state(frame.y, frame.v1, frame.v2)

            r = get_reward(frame, next_frame)
            v = get_v(s)
            next_v = get_v(next_s)

            V[s] = v + alpha * (r + gamma * next_v - v)
            t += 1

V = dict()

data = pd.read_csv('AAPL.csv')

y = data['Close'].to_numpy()
x = np.arange(len(y))

ema14 = ema(y, 14)
ema28 = ema(y, 28)
sample = SampleEpisode()

gen_episode = generator_episodes(y)

# signals = get_signals()

train()

# sorte
V = {k: V[k] for k in sorted(V)}

print(V)
sample.plot()
sample.plot()


plt.plot(x, y)
plt.plot(x, ema14)
plt.plot(x, ema28)
plt.show()