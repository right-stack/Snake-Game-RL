import torch
import random
import numpy as np
from collections import deque
# local imports
from model import DQNet, QTrain
from game import SnakeAI, Direction, Point
from visualize import plot

# Hyper params
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        """
        Initialize Agent
        Params
            - game_num: game number
            - epsilon: for randomness
            - gamma: discount rate
            - memory: a double ended queue to serve as replay buffer
            - model: 
            - trainer: 
        """
        self.game_num = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.98 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # O(1) and popleft()
        self.model = DQNet(11, 256, 3)
        self.trainer = QTrain(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        """
        Gets the state of the game
        State contains eleven values as follows:
        [
            danger straight, danger right, danger left,
            direction left, direction right, direction up, direction down,
            food left, food right, food up, food down
        ]
        """
        head = game.snake[0]
        point_left = Point(head.x - 20, head.y)
        point_right = Point(head.x + 20, head.y)
        point_up = Point(head.x, head.y - 20)
        point_down = Point(head.x, head.y + 20)
        
        go_left = game.direction == Direction.LEFT
        go_right = game.direction == Direction.RIGHT
        go_up = game.direction == Direction.UP
        go_down = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (go_right and game.is_collision(point_right)) or 
            (go_left and game.is_collision(point_left)) or 
            (go_up and game.is_collision(point_up)) or 
            (go_down and game.is_collision(point_down)),

            # Danger right
            (go_up and game.is_collision(point_right)) or 
            (go_down and game.is_collision(point_left)) or 
            (go_left and game.is_collision(point_up)) or 
            (go_right and game.is_collision(point_down)),

            # Danger left
            (go_down and game.is_collision(point_right)) or 
            (go_up and game.is_collision(point_left)) or 
            (go_right and game.is_collision(point_up)) or 
            (go_left and game.is_collision(point_down)),
            
            # Direction
            go_left,
            go_right,
            go_up,
            go_down,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)
        

    def experience(self, state, action, reward, next_state, done):
        """
        Store state, action, reward, next_state and done variables to memory
        """
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def replay_buffer(self):
        """
        The double ended queue that stores our samples 
        """
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.return_dtype(states, actions, rewards, next_states, dones)

    
    # return dtype of the state, action, reward, next_state and done
    def episode(self, state, action, reward, next_state, done):
        self.trainer.return_dtype(state, action, reward, next_state, done)

    def get_action(self, state):
        """
        Give best action based on current state
            - Arg: state
            - Return: max of actions predicted
        """
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 50 - self.game_num
        final_move = [0,0,0]
        # Exploration
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:   # Exploitation
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()  # choose argmax (optimal policy) of predictions. 
            final_move[move] = 1

        return final_move


def train():
    """
    Train DQN agent
    """
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeAI()
    while True:
        # get old state
        old_state = agent.get_state(game)
        
        # get move
        final_move = agent.get_action(old_state)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        new_state = agent.get_state(game)

        # episode
        agent.episode(old_state, final_move, reward, new_state, done)

        # Memory to store experience/episodes
        agent.experience(old_state, final_move, reward, new_state, done)

        if done:
            # replay buffer, plot scores and mean_scores
            game.reset()
            agent.game_num += 1
            agent.replay_buffer()

            if score > record:
                record = score
                agent.model.save()    

            print('No of Games: ', agent.game_num, 'Score of Game: ', score, 'High Score:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.game_num
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()