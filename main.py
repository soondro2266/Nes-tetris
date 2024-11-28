import torch
from torch import nn
import torch.nn.functional as F
import TetrisDataset
from DQN import DQN
from tetris import tetris

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

print("test")
game = tetris()
model = DQN(device)

train_gap = 1
epoch = 20

for i in range(epoch):
    game.reset()
    states:dict = game.get_next_states()
    current_state = [0, 0, 0, 0]
    done = False

    while not done:
        max_q = -1
        best_state = [0, 0, 0, 0]
        best_action = [3, 0]
        for action, state in states.items():
            q = model.predict(state)
            if q > max_q:
                best_state = state
                max_q = q
                best_action = action
        print(best_action)
        score, done = game.play(best_action[0], best_action[1], render=True)
        model.add_memory(current_state, best_state, score, done)
        current_state = best_state

    if i % train_gap == 0:
        print("haha")
        model.train()


print("finish")

torch.save(model.state_dict(), "model.pth")


