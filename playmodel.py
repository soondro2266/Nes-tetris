from DQN import DQN
import torch
from tetris import tetris

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
filename = "modelv2.pth"


model = DQN(device)
model.load_state_dict(torch.load(f"model\\{filename}", weights_only=True))
model.eval()
game = tetris()

for i in range(round):
    game.reset()
    
    current_state = [0, 0, 0, 0]
    done = False
    
    while not done:

        states:dict = game.get_next_states()
        max_q = -1000000
        best_state = [0, 0, 0, 0]
        best_action = [3, 0]
        for action, state in states.items():
            q = model.predict(state)
            if q > max_q:
                best_state = state
                max_q = q
                best_action = action
        score, done = game.play(best_action[0], best_action[1], render=True)
    
    print(f"round {i}, score : {game.score}")