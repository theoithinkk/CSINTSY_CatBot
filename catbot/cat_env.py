import os
import random
import time
from typing import Optional, Tuple, Any, Type
from abc import ABC, abstractmethod
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class Cat(ABC):
    def __init__(self, grid_size: int, tile_size: int):
        self.grid_size = grid_size
        self.tile_size = tile_size
        self.pos = np.zeros(2, dtype=np.int32)
        self.visual_pos = np.zeros(2, dtype=float)
        
        self.player_pos = np.zeros(2, dtype=np.int32)
        self.prev_player_pos = np.zeros(2, dtype=np.int32)
        self.last_player_action = None
        
        self.current_distance = 0  
        self.prev_distance = 0     
        
        self._load_sprite()
    
    @abstractmethod
    def _get_sprite_path(self) -> str:
        pass
    
    def _load_sprite(self):
        img_path = self._get_sprite_path()
        if not os.path.exists(img_path):
            self.sprite = pygame.Surface((self.tile_size, self.tile_size))
            self.sprite.fill((200, 100, 100))
            return
        try:
            self.sprite = pygame.image.load(img_path)
            self.sprite = self.sprite.convert_alpha()
            self.sprite = pygame.transform.scale(self.sprite, (self.tile_size, self.tile_size))
        except Exception as e:
            self.sprite = pygame.Surface((self.tile_size, self.tile_size))
            self.sprite.fill((200, 100, 100))
    
    def update_player_info(self, player_pos: np.ndarray, player_action: int) -> None:
        self.prev_player_pos = self.player_pos.copy()
        self.player_pos = player_pos.copy()
        self.last_player_action = player_action

        self.prev_distance = abs(self.pos[0] - self.prev_player_pos[0]) + abs(self.pos[1] - self.prev_player_pos[1])
        self.current_distance = abs(self.pos[0] - self.player_pos[0]) + abs(self.pos[1] - self.player_pos[1])
    
    def player_moved_closer(self) -> bool:
        return self.current_distance < self.prev_distance
    
    @abstractmethod
    def move(self) -> None:
        pass
    
    def reset(self, pos: np.ndarray) -> None:
        self.pos = pos.copy()
        self.visual_pos = pos.astype(float)
    
    def update_visual_pos(self, dt: float, animation_speed: float) -> None:
        for i in range(2):
            diff = self.pos[i] - self.visual_pos[i]
            if abs(diff) > 0.01:
                self.visual_pos[i] += np.clip(diff * animation_speed * dt, -1, 1)

####################################
# CAT BEHAVIOR IMPLEMENTATIONS     #
####################################

class BatmeowCat(Cat):
    def _get_sprite_path(self) -> str:
        return "images/batmeow-dp.png"
    def move(self) -> None:
        pass

class MittensCat(Cat):
    def _get_sprite_path(self) -> str:
        return "images/mittens-dp.png"
    def move(self) -> None:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        random.shuffle(dirs)
        d = dirs[0]
        new_r = min(max(0, self.pos[0] + d[0]), self.grid_size - 1)
        new_c = min(max(0, self.pos[1] + d[1]), self.grid_size - 1)
        self.pos[0] = new_r
        self.pos[1] = new_c

class PaotsinCat(Cat):
    def _get_sprite_path(self) -> str:
        return "images/paotsin-dp.png"

    def move(self) -> None:
        current_distance = abs(self.pos[0] - self.player_pos[0]) + abs(self.pos[1] - self.player_pos[1])
        if current_distance > 4:
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            random.shuffle(dirs)
            d = dirs[0]
            new_r = min(max(0, self.pos[0] + d[0]), self.grid_size - 1)
            new_c = min(max(0, self.pos[1] + d[1]), self.grid_size - 1)
            self.pos[0] = new_r
            self.pos[1] = new_c
            return
            
        possible_moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        random.shuffle(possible_moves)
        best_move = None
        best_distance = None
        
        for dr, dc in possible_moves:
            new_r = min(max(0, self.pos[0] + dr), self.grid_size - 1)
            new_c = min(max(0, self.pos[1] + dc), self.grid_size - 1)
            
            distance = abs(new_r - self.player_pos[0]) + abs(new_c - self.player_pos[1])
            
            if best_distance is None:
                best_move = (dr, dc)
                best_distance = distance
            elif self.player_moved_closer():
                if distance > best_distance:
                    best_move = (dr, dc)
                    best_distance = distance
            else:
                if distance < best_distance:
                    best_move = (dr, dc)
                    best_distance = distance
        
        if self.player_moved_closer():
            self.pos[0] = min(max(0, self.pos[0] + best_move[0]), self.grid_size - 1)
            self.pos[1] = min(max(0, self.pos[1] + best_move[1]), self.grid_size - 1)
        elif random.random() < 0.65:
            self.pos[0] = min(max(0, self.pos[0] + best_move[0]), self.grid_size - 1)
            self.pos[1] = min(max(0, self.pos[1] + best_move[1]), self.grid_size - 1)
        else:
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            random.shuffle(dirs)
            d = dirs[0]
            new_r = min(max(0, self.pos[0] + d[0]), self.grid_size - 1)
            new_c = min(max(0, self.pos[1] + d[1]), self.grid_size - 1)
            self.pos[0] = new_r
            self.pos[1] = new_c

class PeekabooCat(Cat):
    def _get_sprite_path(self) -> str:
        return "images/peekaboo-dp.png" 
    
    def move(self) -> None:
        is_adjacent = (
            abs(self.pos[0] - self.player_pos[0]) + abs(self.pos[1] - self.player_pos[1]) == 1
        )
        
        if not is_adjacent:
            return
            
        if (self.pos[0] == 0 and self.player_pos[0] == 0 and self.player_pos[1] == self.pos[1] - 1) or (self.pos[1] == 0 and self.player_pos[1] == 0 and self.player_pos[0] == self.pos[0] - 1):
            return
            
        edge_positions = []
        for i in range(self.grid_size):
            edge_positions.extend([
                (0, i), 
                (self.grid_size-1, i), 
                (i, 0),          
                (i, self.grid_size-1)   
            ])
        
        edge_positions = list(set(edge_positions))
        
        safe_positions = []
        for pos in edge_positions:
            if abs(pos[0] - self.player_pos[0]) + abs(pos[1] - self.player_pos[1]) > 1:
                safe_positions.append(pos)
        
        if safe_positions:
            new_pos = random.choice(safe_positions)
            self.pos[0] = new_pos[0]
            self.pos[1] = new_pos[1]

class SquiddyboiCat(Cat):
    def _get_sprite_path(self) -> str:
        return "images/squiddyboi-dp.png"
    
    def move(self) -> None:
        is_adjacent = (
            abs(self.pos[0] - self.player_pos[0]) + abs(self.pos[1] - self.player_pos[1]) == 1
        )
        if not is_adjacent:
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            random.shuffle(dirs)
            
            for d in dirs:
                new_r = min(max(0, self.pos[0] + d[0]), self.grid_size - 1)
                new_c = min(max(0, self.pos[1] + d[1]), self.grid_size - 1)
                
                would_be_adjacent = (
                    abs(new_r - self.player_pos[0]) + abs(new_c - self.player_pos[1]) == 1
                )
                
                if not would_be_adjacent:
                    self.pos[0] = new_r
                    self.pos[1] = new_c
                    return
            return
            
        dr = self.player_pos[0] - self.pos[0]
        dc = self.player_pos[1] - self.pos[1]
        if dr != 0:
            dr = dr // abs(dr)
        if dc != 0:
            dc = dc // abs(dc)
            
        target_r = self.player_pos[0] + 2 * dr
        target_c = self.player_pos[1] + 2 * dc
        
        if (0 <= target_r < self.grid_size and 0 <= target_c < self.grid_size):
            # Three-space hop is possible
            self.pos[0] = target_r
            self.pos[1] = target_c
            return
            
        target_r = self.player_pos[0] + dr
        target_c = self.player_pos[1] + dc
        
        if (0 <= target_r < self.grid_size and 0 <= target_c < self.grid_size):
            self.pos[0] = target_r
            self.pos[1] = target_c
            return
            
        new_r = min(max(0, self.pos[0] - dr), self.grid_size - 1)
        new_c = min(max(0, self.pos[1] - dc), self.grid_size - 1)
        self.pos[0] = new_r
        self.pos[1] = new_c

#####################################
# TRAINER CAT IMPLEMENTATION        #
#####################################
# You can modify the behavior of    #
# this cat to test your learning    #
# algorithm. This cat will not part #
# of the grading.                   #
#####################################

class TrainerCat(Cat):
    """A customizable cat for students to implement and test their own behavior algorithms.
    
    This cat provides access to:
    - self.pos: Current cat position as [row, col]
    - self.player_pos: Current player position
    - self.prev_player_pos: Previous player position
    - self.last_player_action: Last action (0:Up, 1:Down, 2:Left, 3:Right)
    - self.current_distance: Current Manhattan distance to player
    - self.prev_distance: Previous Manhattan distance to player
    - self.grid_size: Size of the grid (e.g., 8 for 8x8 grid)
    
    Helper methods:
    - self.player_moved_closer(): Returns True if player's last move decreased distance
    """

    #   "pursuer"    - walks toward the bot (stalker-like), opposite of evader
    #   "threshold"  - wanders aimlessly, but strategizes  when the bot gets within 3 squares
    #   "warper"     - teleports to a random square every 5 steps
    #   "evader"     - chooses the square thats farthest from the bot (runs away)
    #   "teleporter" - jumps to a random edge square only when the bot is beside it
    #   "still"      - does not move at all
    
    behavior = "pursuer"
    FLEE_RANGE = 3      # for threshold behavior, distance at which the cat starts running away from the bot
    WARP_PERIOD = 5     # for warper behavior, number of steps between warps
    warp_counter = 0    # for warper behavior, counts steps to determine when to warp

    def _get_sprite_path(self) -> str:
        return "images/trainer-dp.png"

    # ----------------------------------------------
    # Helper methods for behavior movements
    # ----------------------------------------------

    def clamp(self, value):
        # keeps the cat within the board's bounds
        return min(max(value, 0), self.grid_size - 1)

    def distance_to_bot(self, row, col):
        # manhattan distance from the cat to the bot
        return abs(row - self.player_pos[0]) + abs(col - self.player_pos[1])

    def step_by(self, row_change, col_change):
        self.pos[0] = self.clamp(self.pos[0] + row_change)
        self.pos[1] = self.clamp(self.pos[1] + col_change)

    def move_random(self):
        # move one step random direction
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        row_change, col_change = random.choice(moves)
        self.step_by(row_change, col_change)

    def move_away_or_toward(self, run_away):
        """cat tries out all four directions and takes the best one.

        If run_away is True the cat picks the square farthest from the bot,
         else it picks the square closest to the bot.
        """
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        random.shuffle(moves)   

        best_move = moves[0]
        best_distance = None

        for row_change, col_change in moves:
            new_row = self.clamp(self.pos[0] + row_change)
            new_col = self.clamp(self.pos[1] + col_change)
            distance = self.distance_to_bot(new_row, new_col)

            if best_distance is None:
                better = True
            elif run_away:
                better = distance > best_distance
            else:
                better = distance < best_distance

            if better:
                best_move = (row_change, col_change)
                best_distance = distance

        self.step_by(best_move[0], best_move[1])

    # ----------------------------------------------
    # 5 added behaviors for testing
    # ----------------------------------------------

    def move_pursuer(self):
        """Always walks toward the bot."""
        self.move_away_or_toward(run_away=False)

    def move_threshold(self):
        """Ignores the bot until it gets close, then runs."""
        if self.current_distance <= self.FLEE_RANGE:
            self.move_away_or_toward(run_away=True)
        else:
            self.move_random()

    def move_warper(self):

        self.warp_counter += 1

        if self.warp_counter % self.WARP_PERIOD == 0: # every 5 steps, jump to a random square
            self.pos[0] = random.randrange(self.grid_size)
            self.pos[1] = random.randrange(self.grid_size)

    def move_evader(self):
        self.move_away_or_toward(run_away=True)

    def move_teleporter(self):
    
        if self.current_distance != 1: # cat stays put if not adjacent to the bot
            return
        
        edge_squares = [] # every square along the four edges
        for i in range(self.grid_size):
            edge_squares.append((0, i))
            edge_squares.append((self.grid_size - 1, i))
            edge_squares.append((i, 0))
            edge_squares.append((i, self.grid_size - 1))


        safe_squares = [] # from the edge squares, only keep those that are not adjacent to the bot
        for row, col in edge_squares:
            if self.distance_to_bot(row, col) > 1:
                safe_squares.append((row, col))

        if safe_squares:
            self.pos[0], self.pos[1] = random.choice(safe_squares)

    def move(self) -> None:
        if self.behavior == "pursuer":
            self.move_pursuer()
        elif self.behavior == "threshold":
            self.move_threshold()
        elif self.behavior  == "warper":
            self.move_warper()
        elif self.behavior == "evader":
            self.move_evader()
        elif self.behavior == "teleporter":
            self.move_teleporter()
        # "still" and anything else means the cat does not move
        return

#######################################
# END OF CAT BEHAVIOR IMPLEMENTATIONS #
#######################################

class CatChaseEnv(gym.Env):
    """Simple 8x8 grid world where an agent tries to catch a randomly moving cat.

    Observation: Dict with 'agent' and 'cat' positions as (row, col) each in [0..7].
    Action space: Discrete(4) -> 0:Up,1:Down,2:Left,3:Right
    Reward: +1 when agent catches cat (episode ends), -0.01 per step to encourage speed.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, grid_size: int = 8, tile_size: int = 64, cat_type: str = "peekaboo"):
        super().__init__()
        self.grid_size = grid_size
        self.tile_size = tile_size

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Discrete(grid_size * 1000 + grid_size * 100 + grid_size * 10 + grid_size)

        pygame.init()
        self.screen = None
        self.clock = pygame.time.Clock()

        temp_surface = pygame.display.set_mode((1, 1))
        
        agent_img_path = "images/catbot.png"
        if not os.path.exists(agent_img_path):
            print(f"Warning: Agent image file not found: {agent_img_path}")
            self.agent_sprite = pygame.Surface((self.tile_size, self.tile_size))
            self.agent_sprite.fill((100, 200, 100))
        else:
            try:
                self.agent_sprite = pygame.image.load(agent_img_path)
                self.agent_sprite = self.agent_sprite.convert_alpha()
                self.agent_sprite = pygame.transform.scale(self.agent_sprite, (self.tile_size, self.tile_size))
                print(f"Successfully loaded agent sprite: {agent_img_path}")
            except Exception as e:
                print(f"Error loading agent sprite {agent_img_path}: {str(e)}")
                self.agent_sprite = pygame.Surface((self.tile_size, self.tile_size))
                self.agent_sprite.fill((100, 200, 100))

        cat_types = {
            "batmeow": BatmeowCat,          
            "mittens": MittensCat,
            "paotsin": PaotsinCat,
            "peekaboo": PeekabooCat,
            "squiddyboi": SquiddyboiCat,
            "trainer": TrainerCat
        }
        if cat_type not in cat_types:
            raise ValueError(f"Unknown cat type: {cat_type}. Available types: {list(cat_types.keys())}")
        
        self.cat = cat_types[cat_type](grid_size, tile_size)
        
        pygame.display.quit()
        
        self.agent_pos = np.zeros(2, dtype=np.int32)
        self.agent_visual_pos = np.zeros(2, dtype=float)
        self.animation_speed = 48.0  
        self.last_render_time = time.time()
        
        self.agent_bump_offset = np.zeros(2, dtype=float)  
        self.cat_bump_offset = np.zeros(2, dtype=float)    
        self.bump_magnitude = 0.15  
        self.bump_spring = 8.0      
        
        self.done = False

    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> Tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.agent_pos = np.array([0, 0], dtype=np.int32)
        cat_start_pos = np.array([self.grid_size-1, self.grid_size-1], dtype=np.int32)
        self.cat.reset(cat_start_pos)

        self.agent_visual_pos = self.agent_pos.astype(float)
        self.last_render_time = time.time()
        
        self.done = False
        return self._get_obs(), {}

    def _get_obs(self):
        return int(self.agent_pos[0] * 1000 + self.agent_pos[1] * 100 + self.cat.pos[0] * 10 + self.cat.pos[1])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self.done:
            return self._get_obs(), 0.0, True, False, {}
        reward = 0

        old_pos = self.agent_pos.copy()
        
        if action == 0:  
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
            if self.agent_pos[0] == old_pos[0]:  
                self.agent_bump_offset[0] = -self.bump_magnitude
        elif action == 1:  
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
            if self.agent_pos[0] == old_pos[0]:  
                self.agent_bump_offset[0] = self.bump_magnitude
        elif action == 2:  
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
            if self.agent_pos[1] == old_pos[1]:  
                self.agent_bump_offset[1] = -self.bump_magnitude
        elif action == 3:  
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
            if self.agent_pos[1] == old_pos[1]:  
                self.agent_bump_offset[1] = self.bump_magnitude

        info = {}

        if np.array_equal(self.agent_pos, self.cat.pos):
            self.done = True
            return self._get_obs(), reward, True, False, info

        self.cat.update_player_info(self.agent_pos, action)
        self.cat.move()

        if np.array_equal(self.agent_pos, self.cat.pos):
            self.done = True

        return self._get_obs(), float(reward), bool(self.done), False, info

    def render(self, mode: str = "human"):
        if self.screen is None:
            width = self.grid_size * self.tile_size
            height = self.grid_size * self.tile_size
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Cat Chase")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        self.screen.fill((25, 48, 15))

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                rect = pygame.Rect(c * self.tile_size, r * self.tile_size, self.tile_size, self.tile_size)
                color = (35, 61, 20) if (r + c) % 2 == 0 else (25, 48, 15)
                pygame.draw.rect(self.screen, color, rect)

        current_time = time.time()
        dt = current_time - self.last_render_time
        self.last_render_time = current_time
        
        for i in range(2):
            diff = self.agent_pos[i] - self.agent_visual_pos[i]
            if abs(diff) > 0.01:
                self.agent_visual_pos[i] += np.clip(diff * self.animation_speed * dt, -1, 1)
            
            if abs(self.agent_bump_offset[i]) > 0.001:
                self.agent_bump_offset[i] *= max(0, 1 - self.bump_spring * dt)
        
        self.cat.update_visual_pos(dt, self.animation_speed)
        
        old_cat_pos = self.cat.pos.copy()
        if not np.array_equal(old_cat_pos, self.cat.pos):
            for i in range(2):
                if old_cat_pos[i] == self.cat.pos[i] and (
                    (old_cat_pos[i] == 0 and self.cat.pos[i] == 0) or 
                    (old_cat_pos[i] == self.grid_size - 1 and self.cat.pos[i] == self.grid_size - 1)
                ):
                    self.cat_bump_offset[i] = self.bump_magnitude if old_cat_pos[i] == self.grid_size - 1 else -self.bump_magnitude
        
        for i in range(2):
            if abs(self.cat_bump_offset[i]) > 0.001:
                self.cat_bump_offset[i] *= max(0, 1 - self.bump_spring * dt)
        
        cat_x = (self.cat.visual_pos[1] + self.cat_bump_offset[1]) * self.tile_size
        cat_y = (self.cat.visual_pos[0] + self.cat_bump_offset[0]) * self.tile_size
        cat_rect = pygame.Rect(cat_x, cat_y, self.tile_size, self.tile_size)
        self.screen.blit(self.cat.sprite, cat_rect)

        ag_x = (self.agent_visual_pos[1] + self.agent_bump_offset[1]) * self.tile_size
        ag_y = (self.agent_visual_pos[0] + self.agent_bump_offset[0]) * self.tile_size
        ag_rect = pygame.Rect(ag_x, ag_y, self.tile_size, self.tile_size)
        self.screen.blit(self.agent_sprite, ag_rect)

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.screen = None
        pygame.quit()

def make_env(cat_type: str = "batmeow"):
    return CatChaseEnv(cat_type=cat_type)