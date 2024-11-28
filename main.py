import torch
from torch import nn
import torch.nn.functional as F
import TetrisDataset
from DQN import DQN
from tetris import tetris
import matplotlib.pyplot as plt
import random

device = torch.device("cuda")
print(device)

print("test")


train_gap = 1
render_gap = 1
epoch = 10
all_score = []
batch_size = 128
epsilon = 0.9

game = tetris()
model = DQN(device, batch_size=batch_size)


for i in range(epoch):
    game.reset()
    states:dict = game.get_next_states()
    current_state = [0, 0, 0, 0]
    done = False
    to_render = False
    if i % render_gap == 0:
        to_render = True
    
    while not done:
        """
        if random.random() <= epsilon-(0.8*i/epoch):
            best_action = random.choice(list(states))
            best_state = states[best_action]
        else:"""
        max_q = -1000000
        best_state = [0, 0, 0, 0]
        best_action = [3, 0]
        for action, state in states.items():
            q = model.predict(state)
            if q > max_q:
                best_state = state
                max_q = q
                best_action = action

        score, done = game.play(best_action[0], best_action[1], render=to_render)


        model.add_memory(current_state, best_state, score, done)
        current_state = best_state

    all_score.append(game.score)

    if i % train_gap == 0:
        model.train()


print("finish")

plt.plot([i+1 for i in range(len(all_score))], all_score, 'r', linestyle='solid', label = 'train')
plt.show()

torch.save(model.state_dict(), "model.pth")


