import torch
from torch import nn
import torch.nn.functional as F
import TetrisDataset
from DQN import DQN
from tetris import tetris
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
filename = "modelv2.pth"
save_model = True

train_gap = 1
render_gap = 100
epoch = 2000
all_score = []
batch_size = 128
epsilon_start = 1
epsilon_end = 0
epsilon_end_epoch = 1000
add_score_gap = 10

max_score = -1
best_score = -1
end_score = 20000

game = tetris(end_score)
model = DQN(device, batch_size=batch_size)

for i in tqdm(range(epoch)):
    game.reset()
    
    current_state = [0, 0, 0, 0]
    done = False
    to_render = False
    if i % render_gap  == 0 and i > 800:
        to_render = True
    
    while not done:

        states:dict = game.get_next_states()
        if random.random() <= max(epsilon_end, epsilon_start - 1.2*(epsilon_start - epsilon_end)*(i / epsilon_end_epoch)):
            best_action = random.choice(list(states))
            best_state = states[best_action]
        else:
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

    max_score = max(max_score, game.score)
    
    if i % train_gap == 0:
        model.train_dqn()

    if i % add_score_gap == 0:
        all_score.append(max_score)
        max_score = -1
    
    if save_model and game.score > best_score:
        torch.save(model.state_dict(), f"model\\{filename}")
        print(f"save model for {game.score} point.")
        best_score = max(best_score, game.score)
    
    if game.score >= end_score:
        break
    
all_score.append(best_score)
plt.plot([i+1 for i in range(len(all_score))], all_score, 'r', linestyle='solid', label = 'train')
plt.show()