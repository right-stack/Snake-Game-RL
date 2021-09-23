# DQN-Snake
The classic Snake game using Deep Q-Networks (https://arxiv.org/pdf/1509.06461.pdf). The snake game environment used was built with Pygame. A DQN agent plays the game by going towards the objective, the action is predicted by the Neural Network from the state.

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

The langage that is used is Python (version 3.8.5), which can be downloaded at https://www.python.org/. 

[Pytorch](https://www.pytorch.org/), an open source deep learning library. 


The other libraries used are in the file ```requirements.txt```
```
pip install -r requirements.txt
```
