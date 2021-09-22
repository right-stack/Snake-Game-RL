# Implementation of Snake game 
Here we implemented the snake game using Deep Q learning model (https://arxiv.org/pdf/1509.06461.pdf). We started by building the environment which includes snake, food,playground or screen and set some policies of the game. Some of those policies are like to stop a game if the snake collides with the border of the screen or hit itself. Our model as an agent allowed to start playing randomly by interacting with the environment so that after some games it can manage to play well. The more games the more our agent plays well.

# Installation

pip install torch

pip install pygame


# Run code

cd Snake-RL-AMMI

python dqn-agent.py


# Create the screen
We used pygame to create the screen using display.set_ mode() package and fill it with RGB color(white). The screen we used has a rectangle shape with the width= 480 and lenght = 640. The screen is designed in such a way that you can update the changes that are made like snake movement.

In implementation we started initializing pygame using init() and and quit() at the end of the code. we have also an update method "update_ui()", where we used flip() to update the complete screen.


# Moving the snake



# Game over



# Adding snake Food


# State


# Action


# Display results

