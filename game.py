# This environment is an adaptation of snake game made with pygame from https://github.com/python-engineer/python-fun 

import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple

# Initialize pygame and specify font, font = Arial
pygame.init()
font = pygame.font.Font('font.ttf', 20)

# Create enum class to make direction easy to read and write
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# This will allow us to assign meaning to position of the snake tuple.
# Allow to access fields by name instead of position index   
Point = namedtuple('Point', 'x, y')

# RGB for game visualization
WHITE = (255, 255, 255)
FOODCOLOR = (0,100,10)
SNAKE_BODY = (0, 50, 0)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 20

class SnakeAI: 
    def __init__(self, w=640, h=480):
        """ Initialize agent.
        Params
            - w: number of states available to the agent
            - h: number of actions available to the agent
        """
        self.w = w
        self.h = h
        # init display, caption and track time
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('SNAKE')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        """
        After game over, agent initialize and start a new game
        """
        # initial snake direction
        self.direction = Direction.RIGHT
        # snake is a list of head and two other points (values)
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        # Initialize game variables
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        
    def _place_food(self):
        """
        Place the food at randomly generated x and y axis
        """
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        # If food generated point is on snake body, place food again
        if self.food in self.snake:
            self._place_food()
        
        food = self.food
        head = self.head

    def food_distance(self):
        """
        Computes distance from head to food
            - Args: self.food, self.head
            - Returns: distance from head to food
        Intend to use this to compute feedback signal from environment for additional reward for SARSA
        """
        food = list(self.food)
        head = list(self.head)
        food_distance = [a_i - b_i for a_i, b_i in zip(food, head)]
        return food_distance

        
    def play_step(self, action):
        """
        Game iteration.
        Args:
            - action (s)
        Results:
            - If game_over, score and reward
            - If !game_over, move (update frames)
            - If snake gets food, update score, reward and place food
        """
        self.frame_iteration += 1
        # 1. Game Quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        reward = 0
        # 3. If game_over, return score and reward
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
            
        # 4. If snake gets food, update score, reward and place food
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        """
        Consequences of where the head moves to
        - If none, then the point becomes snake head
        - If boundary, is_collision = True
        - If snake body, is_collision = True
        """
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False
        
    def _update_ui(self):
        """
        Display parameters for background, snake, food and Score updates
        """
        self.display.fill(WHITE)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, SNAKE_BODY, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, SNAKE_BODY, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.ellipse(self.display, FOODCOLOR, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, BLACK)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, action):
        """
        Using clockwise movement, manipulate index to change directions
        """
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        indx = clockwise.index(self.direction)
        # Actions to predict [1, 0, 0]: straight, [0, 1, 0]: right, [0, 0, 1]: left
        if np.array_equal(action, [1, 0, 0]):
            new_direction = clockwise[indx] # Go straight
        elif np.array_equal(action, [0, 1, 0]):
            next_index = (indx + 1) % 4
            new_direction = clockwise[next_index] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_index = (indx - 1) % 4    # Index 0
            new_direction = clockwise[next_index] # left turn r -> u -> l -> d

        self.direction = new_direction

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
            