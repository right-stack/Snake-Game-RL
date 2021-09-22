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


# Create and Move the snake
To create a snake we draw three stacked rectangles stand for head, body and tail. we used pygame.draw.rect() and fill it with desired color and size. The length of the snake is contained in a list and it increases when the snake eats the food. We allowed the snake to move clockwise in different directions(up, down, left and right) within the screen.


# Game over

We set that the game will end if the snake hits itself or the boundaries of the screen. After that we reset again and start a new game.


# Adding snake Food

We designed the food of snake to have a shape of circle filled with the color(e.g: Green). The food is randomly placed in a screen on a position (x,y), so that the snake can find it while it is moving.


# Display results
