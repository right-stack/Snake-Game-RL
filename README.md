# DQN-Snake
The classic Snake game played by a Deep Q-Networks agent (https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf). The snake game environment used was implemented in Pygame. The DQN agent plays the game by going towards the objective, the action is predicted by the Neural Network from the state.

## Usage

To download the repository, open a cmd prompt and execute 
```
git clone https://github.com/right-stack/Snake-RL-AMMI.git
```

This will create a folder on your computer that you can access from the command prompt by executing:

```
cd Snake-RL-AMMI

python dqn-agent.py
``` 

## Requirements

Programming language used is [Python](https://www.python.org/) (version 3.8.5). 

[Pytorch](https://www.pytorch.org/), an open source deep learning library. 


The other libraries used are in the file ```requirements.txt```
```
pip install -r requirements.txt
```
