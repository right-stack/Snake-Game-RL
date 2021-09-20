import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class DQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize DQN class
            Params
                - input layer size
                - hidden layer size
                - output layer size
        """
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='dqn-model.pth'):
        """
        Save model as dqn-model.pth in model folder path
        """
        model_folder_path = './models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrain:
    def __init__(self, model, lr, gamma):
        """
        Initialize Qtrain class
            Params
                - learning rate
                - gamma/discount factor
                - model
                - optimizer 
                - criterion
        """
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def return_dtype(self, state, action, reward, next_state, done):
        """
        Gets the policy using epsilon-greedy algorithm
        Args:
            - state: the state of the agent
            - action: current action
            - reward
            - next state
            - done or not 
        Returns:
            - manipulates the dtype of params to tensor of floating point values 
        """
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred_q = self.model(state)

        target = pred_q.clone()
        for i in range(len(done)):
            Q_new = reward[i]
            if not done[i]:
                # Q_new = reward + discount factor * max(next_predicted Q values)
                Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))

            target[i][torch.argmax(action[i]).item()] = Q_new
    
        # Optimizer, loss function and backprop
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred_q)
        loss.backward()

        self.optimizer.step()