import torch
from torch import nn
import torch.nn.functional as F
from TetrisDataset import TetrisDataset


class DQN(nn.Module):

    def __init__(self, device, state_dim=4, action_dim=1, batch_size=10, lr = 0.005, discount = 0.95):
        super(DQN, self).__init__()
        self.loss = []
        self.discount = discount
        self.memory = []
        self.batch_size = batch_size
        self.device = device
        self.lr = lr
        self.criterion = torch.nn.MSELoss()
        self.L1 = nn.Linear(state_dim, 128, device=device)
        self.L2 = nn.Linear(128, 32, device=device)
        self.L3 = nn.Linear(32, 8, device=device)
        self.L4 = nn.Linear(8, action_dim, device=device)

    def predict(self, x:list):
        x_t = (torch.tensor(x, device=self.device, dtype=torch.float32).unsqueeze(dim = 0))
        x_f = self.forward(x_t)
        return x_f.item()

    def forward(self, x)-> torch.Tensor:
        x1 = F.relu(self.L1(x))
        x2 = F.relu(self.L2(x1))
        x3 = F.relu(self.L3(x2))
        return self.L4(x3)

    def add_memory(self, state, best_state, reward, done):
        self.memory.append([state, best_state, reward, done])


    def train_dqn(self):# 1 episode
        x = [data[0] for data in self.memory]
        y = []
        for data in self.memory:
            if data[3] == 1:
                y.append(data[2])
            else:
                y.append(data[2]+self.discount * self.predict(data[1]))
        ds = TetrisDataset(x, y, self.device)
        iterator = torch.utils.data.DataLoader(ds, self.batch_size, shuffle = True)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        total_loss = 0.0
        length = 0
        for x_batch, y_batch in iterator:
            optimizer.zero_grad()
            y_prediction = self.forward(x_batch)
            loss:torch.Tensor = self.criterion(y_prediction, y_batch.unsqueeze(dim = 1))
            loss.backward()
            optimizer.step()
            length += 1
            total_loss += loss
        self.loss.append(total_loss/length)
        #clear memory
        self.memory = []