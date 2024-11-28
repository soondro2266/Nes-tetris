import torch
from torch import nn
import torch.nn.functional as F
from TetrisDataset import TetrisDataset



class DQN(nn.Module):

    def __init__(self, device, state_dim=4, action_dim=1, hidden_dim = 32, batch_size=10, lr = 0.01):
        super().__init__()
        #self.loss = []
        self.memory = []
        self.batch_size = batch_size
        self.device = device
        self.lr = lr
        self.criterion = torch.nn.MSELoss()
        self.L1 = nn.Linear(state_dim, hidden_dim, device=device)
        self.L2 = nn.Linear(hidden_dim, hidden_dim, device=device)
        self.L3 = nn.Linear(hidden_dim, action_dim, device=device)

    def predict(self, x:list):
        x_t = (torch.tensor(x, device=self.device, dtype=torch.float32).unsqueeze(dim = 0))
        x2 = self.forward(x_t)
        return x2.item()

    def forward(self, x)-> torch.Tensor:
        x1 = F.relu(self.L1(x))
        x2 = F.relu(self.L2(x1))
        return self.L3(x2)

    def add_memory(self, state, best_state, reward, done):
        self.memory.append([state, best_state, reward, done])

    def train(self):# 1 episode
        n = len(self.memory)
        x = [data[0] for data in self.memory]
        y = []
        for data in self.memory:
            if data[3] == 1:
                y.append(data[2])
            else:
                y.append(data[2]+self.predict(data[1]))
        ds = TetrisDataset(x, y, self.device)
        iterator = torch.utils.data.DataLoader(ds, self.batch_size, shuffle = True)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        for x_batch, y_batch in iterator:
            optimizer.zero_grad()
            y_prediction = self.forward(x_batch)
            loss:torch.Tensor = self.criterion(y_prediction, y_batch.unsqueeze(dim = 1))
            #self.loss.append(torch.mean(loss).item())
            loss.backward()
            optimizer.step()
        #clear memory
        self.memory = []