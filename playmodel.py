from DQN import DQN
import torch
from tetris import tetris

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
filename = "modelv1498.pth"


model = DQN(device)
model.load_state_dict(torch.load(f"model\\{filename}", weights_only=True))
model.eval()
game = tetris(1000)

for i in range(50):
    game.reset()
    
    current_state = [0, 0, 0, 0]
    done = False
    
    while not done:

        states:dict = game.get_next_states()
        max_value = -1000000
        best_state = [0, 0, 0, 0]
        best_action = [3, 0]
        for action, state in states.items():
            value = 0
            if state[2]:
                value = state[1]
            else:
                q = model.predict(state[0])
                value = state[1]+0.95*q
                
            if value > max_value:
                best_state = state[0]
                max_value = value
                best_action = action

        score, done = game.play(best_action[0], best_action[1], render=True)
    
    print(f"round {i+1}, score : {game.score}")