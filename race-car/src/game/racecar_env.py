# src/game/racecar_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

# --- MODIFIED IMPORT ---
# Import the entire core module instead of its individual components
from . import core

# Define the actions the agent can take
ACTIONS = ['ACCELERATE', 'DECELERATE', 'STEER_LEFT', 'STEER_RIGHT', 'NOTHING']
ACTION_MAP = {i: action for i, action in enumerate(ACTIONS)}

class RaceCarEnv(gym.Env):
    """A custom Gymnasium environment for the Race Car game."""
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        # Define action and observation space
        self.action_space = spaces.Discrete(len(ACTIONS))

        # Observations are 16 sensor readings + 2 velocity components
        num_observations = 18 # 16 sensors + vx + vy
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(num_observations,), dtype=np.float32)

    def _get_obs(self):
        """Extracts an observation vector from the game state."""
        # --- MODIFIED TO USE core.STATE ---
        if core.STATE is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        ego = core.STATE.ego
        
        sensor_readings = []
        for sensor in core.STATE.sensors:
            reading = sensor.reading if sensor.reading is not None else 1000.0
            sensor_readings.append(reading / 1000.0)

        velocity_x = ego.velocity.x / 20.0
        velocity_y = ego.velocity.y / 2.0

        obs = np.array(sensor_readings + [velocity_x, velocity_y], dtype=np.float32)
        return np.clip(obs, -1.0, 1.0)

    def _get_info(self):
        """Returns diagnostic information."""
        # --- MODIFIED TO USE core.STATE ---
        return {"distance": core.STATE.distance, "ticks": core.STATE.ticks}

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed)

        seed_value = str(seed) if seed is not None else None
        
        # This will now correctly use a different seed for each new episode
        core.initialize_game_state(api_url="local", seed_value=seed_value)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        try:
            action_int = int(np.asarray(action).squeeze())
        except Exception as e:
            raise ValueError(f"Invalid action format {type(action)}: {action}") from e
        if action_int not in ACTION_MAP:
            action_int = ACTIONS.index('NOTHING')
        action_string = ACTION_MAP[action_int]

        prev_distance = core.STATE.distance

        core.update_game(action_string)
        core.STATE.ticks += 1

        crashed = False
        ego = core.STATE.ego
        for car in core.STATE.cars:
            if car is not ego and core.intersects(ego.rect, car.rect):
                crashed = True
                break
        if not crashed:
            for wall in core.STATE.road.walls:
                if core.intersects(ego.rect, wall.rect):
                    crashed = True
                    break
        core.STATE.crashed = core.STATE.crashed or crashed

        observation = self._get_obs()

        if crashed:
            reward = -100.0
        else:
            sensor_obs = observation[:16]

            FWD_CONE   = (7, 9, 0, 10, 1)
            LEFT_CONE  = (8, 7, 9)             
            RIGHT_CONE = (10, 1, 11)          

            delta_dist = core.STATE.distance - prev_distance        # ~ ego.vx per tick
            fwd_min = float(np.min(sensor_obs[list(FWD_CONE)]))     # 0..1 (1 = far)
            danger_gate = float(np.clip((fwd_min - 0.45) / 0.20, 0.0, 1.0))
            progress_r = 0.1 * max(delta_dist, 0.0) * danger_gate

            MAX_SENSOR_PX = 1000.0  # sensor reach (px)
            FPS = 60.0
            vx = max(core.STATE.ego.velocity.x, 0.0)  # px/tick

            d_fwd_px = fwd_min * MAX_SENSOR_PX
            ttc_fwd = (d_fwd_px / max(vx, 1e-6)) / FPS
            TTC_CUTOFF_FWD = 1.8
            TTC_SCALE_FWD  = 4.5
            danger_fwd = float(np.clip((TTC_CUTOFF_FWD - ttc_fwd) / TTC_CUTOFF_FWD, 0.0, 1.0))
            pen_ttc_fwd = (danger_fwd ** 2) * TTC_SCALE_FWD

            left_min  = float(np.min(sensor_obs[list(LEFT_CONE)]))
            right_min = float(np.min(sensor_obs[list(RIGHT_CONE)]))
            ttc_left  = (left_min * MAX_SENSOR_PX)  / max(vx, 1e-6) / FPS
            ttc_right = (right_min * MAX_SENSOR_PX) / max(vx, 1e-6) / FPS
            TTC_CUTOFF_SIDE = 1.2
            TTC_SCALE_SIDE  = 3.0
            danger_left  = float(np.clip((TTC_CUTOFF_SIDE - ttc_left) / TTC_CUTOFF_SIDE, 0.0, 1.0))
            danger_right = float(np.clip((TTC_CUTOFF_SIDE - ttc_right) / TTC_CUTOFF_SIDE, 0.0, 1.0))
            pen_ttc_side = (danger_left ** 2 + danger_right ** 2) * (TTC_SCALE_SIDE * 0.5)

            if action_string == 'STEER_LEFT' and left_min < (fwd_min + 0.10):
                steer_into_danger += 0.08 + 0.8 * max(0.0, 0.5 - left_min)
            if action_string == 'STEER_RIGHT' and right_min < (fwd_min + 0.10):
                steer_into_danger += 0.08 + 0.8 * max(0.0, 0.5 - right_min)

            act_shaping = 0.0
            if action_string == 'ACCELERATE' and danger_fwd > 0.35:
                act_shaping += 0.7 * (danger_fwd - 0.35)   # was 0.5
            elif action_string == 'DECELERATE' and max(danger_fwd, danger_left, danger_right) > 0.35:
                act_shaping -= 0.35 * (max(danger_fwd, danger_left, danger_right) - 0.35)

            headway_sec = ttc_fwd
            HEADWAY_TARGET = 2.0
            headway_bonus = 0.08 * np.clip((headway_sec - HEADWAY_TARGET) / HEADWAY_TARGET, 0.0, 1.0)
            headway_bonus *= float(np.clip(vx / 12.0, 0.0, 1.0)) * danger_gate

            turn_pen = 0.02 if action_string in ('STEER_LEFT', 'STEER_RIGHT') else 0.0

            survival = 0.01
            reward = (
                progress_r
                - pen_ttc_fwd
                - pen_ttc_side
                - steer_into_danger
                - act_shaping
                - turn_pen
                + headway_bonus
                + survival
            )

        terminated = crashed
        truncated = core.STATE.ticks >= core.MAX_TICKS

        info = self._get_info()
        if not crashed:
            info.update({
                "delta_distance": float(core.STATE.distance - prev_distance),
                "fwd_min": float(fwd_min),
                "left_min": float(left_min),
                "right_min": float(right_min),
                "ttc_fwd": float(ttc_fwd),
                "ttc_left": float(ttc_left),
                "ttc_right": float(ttc_right),
                "headway_sec": float(headway_sec),
                "vx": float(vx),
            })
        else:
            info.update({
                "delta_distance": float(core.STATE.distance - prev_distance),
                "fwd_min": 0.0,
                "left_min": 0.0,
                "right_min": 0.0,
                "ttc_fwd": 0.0,
                "ttc_left": 0.0,
                "ttc_right": 0.0,
                "headway_sec": 0.0,
                "vx": float(core.STATE.ego.velocity.x),
            })

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info


    def render(self):
        if self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        # --- MODIFIED TO USE core constants ---
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((core.SCREEN_WIDTH, core.SCREEN_HEIGHT))
            pygame.display.set_caption("Race Car Game - SB3")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.screen.fill((0, 0, 0))
        # --- MODIFIED TO USE core.STATE ---
        self.screen.blit(core.STATE.road.surface, (0, 0))
        for wall in core.STATE.road.walls:
            wall.draw(self.screen)
        for car in core.STATE.cars:
            if car.sprite:
                self.screen.blit(car.sprite, (car.x, car.y))
        for sensor in core.STATE.sensors:
            sensor.draw(self.screen)
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()