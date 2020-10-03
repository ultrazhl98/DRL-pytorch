import argparse
from itertools import count
import torch
import torch.nn as nn 
from torch.distributions import Categorical
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import gym 


# parameter
parser = argparse.ArgumentParser(description="policy gradient parameter")
parser.add_argument('--lr', type=float, default=0.01,
                    help='learing rate')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--num_episodes', type=int, default=500,
                    help='num episode')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help="discount factor (default:0.99)")
parser.add_argument('--seed', type=int, default=543, metavar='N', 
                    help='random seed (default:543)')
parser.add_argument('--render', default='False', action='store_true',
                    help='render the environment')      
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default:10)')
args = parser.parse_args()

# create environment
env = gym.make('CartPole-v0')
state_space = env.observation_space.shape[0]
action_space = env.action_space.n


def plot_durations(episode_durations):
    plt.ion()  # 进入交互模式
    plt.figure(2)  
    plt.clf()  # 清除所有轴
    duration_t = torch.FloatTensor(episode_durations)
    plt.title('Training')
    plt.xlabel('Episodes')
    plt.ylabel('Duration')
    plt.plot(duration_t.numpy())

    if len(duration_t) >= 100:
        means = duration_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.00001)


# create policy
class Policy(nn.Module):
    def __init__(self, state_space, action_space):
        super(Policy, self).__init__()
        self.state_space = state_space
        self.action_space = action_space

        self.fc1 = nn.Linear(self.state_space, 128)
        self.fc2 = nn.Linear(128, self.action_space)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.softmax(x, dim=-1)

        return x


# create policy
policy = Policy(state_space, action_space)         
optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)


# train policy
def train():

    episode_durations = []
    state_batch = []
    action_batch = []
    reward_batch = []
    steps = 0

    for episode in range(args.num_episodes):
        state = env.reset()
        state = torch.from_numpy(state).float()

        if args.render:        
            env.render()

        for t in count():
            # choose action
            prob = policy(state)
            categorical = Categorical(prob)
            action = categorical.sample()

            action = action.detach().numpy().astype('int32')
            next_state, reward, done, _ = env.step(action)
            reward = 0 if done else reward 
            if args.render:
                env.render()

            state_batch.append(state)
            action_batch.append(float(action))
            reward_batch.append(reward)

            state = next_state
            state = torch.from_numpy(state).float()

            steps += 1
            if done:
                episode_durations.append(t+1)                
                plot_durations(episode_durations)
                break

        # update policy
        if episode > 0 and episode % args.batch_size == 0:

            r = 0
            for i in reversed(range(steps)):
                if reward_batch[i] == 0:
                    r = 0
                else:
                    r = r * args.gamma + reward_batch[i]
                    reward_batch[i] = r 

            # normalize reward
            reward_mean = np.mean(reward_batch)
            reward_std = np.std(reward_batch)
            reward_batch = (reward_batch - reward_mean) / reward_std

            # gradient desent
            optimizer.zero_grad()

            for i in range(steps):
                state = state_batch[i]
                action = torch.Tensor([action_batch[i]])
                reward = reward_batch[i]

                probs = policy(state)
                categorical = Categorical(probs)
                loss = -categorical.log_prob(action) * reward
                loss.backward()
            optimizer.step()

            # clear batch
            state_batch = []
            action_batch = []
            reward_batch = []
            steps = 0


if __name__ == "__main__":
    train()