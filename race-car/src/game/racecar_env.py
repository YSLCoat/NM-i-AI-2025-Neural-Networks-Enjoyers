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

        self._last_back_min = 1.0

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
        """One environment step with TTC-aware shaping + rear-safety shaping."""
        # 1) Robust action handling
        try:
            action_int = int(np.asarray(action).squeeze())
        except Exception as e:
            raise ValueError(f"Invalid action format {type(action)}: {action}") from e
        if action_int not in ACTION_MAP:
            action_int = ACTIONS.index('NOTHING')
        action_string = ACTION_MAP[action_int]

        # 2) Cache progress baseline
        prev_distance = core.STATE.distance

        # 3) Simulate one tick
        core.update_game(action_string)
        core.STATE.ticks += 1

        # 4) Collision detection (+ where it came from)
        crashed = False
        ego = core.STATE.ego
        collider = None
        for car in core.STATE.cars:
            if car is not ego and core.intersects(ego.rect, car.rect):
                crashed = True
                collider = None
                break
        if not crashed:
            for wall in core.STATE.road.walls:
                if core.intersects(ego.rect, wall.rect):
                    crashed = True
                    break
        core.STATE.crashed = core.STATE.crashed or crashed

        # 5) Observation (next state)
        observation = self._get_obs()

        crash_location = "unknown"

        # 6) Reward shaping
        if crashed:
            collided_from_rear = False
            if collider is not None:
                try:
                    collided_from_rear = (collider.rect.centerx < ego.rect.centerx - 2)
                except Exception:
                    collided_from_rear = False
            base = -100.0
            reward = base - (40.0 if collided_from_rear else 0.0)

            # --- NEW: Logic to find crash location ---
            # The observation holds the sensor data from the moment of the crash
            sensor_readings = observation[:16] 
            # Find the index of the sensor with the lowest value (closest to the object)
            closest_sensor_idx = np.argmin(sensor_readings)

            # You'll need a map from the sensor index to its name
            # This must match the order in your README.md
            sensor_names = [
                "left_side", "left_side_front", "left_front", "front_left_front", 
                "front", "front_right_front", "right_front", "right_side_front",
                "right_side", "right_side_back", "right_back", "back_right_back",
                "back", "back_left_back", "left_back", "left_side_back"
            ]
            # Check if the collision was very close (sensor reading near zero)
            if sensor_readings[closest_sensor_idx] < 0.05:
                crash_location = sensor_names[closest_sensor_idx]
            # Prepare defaults for logging below
            fwd_min = left_min = right_min = 0.0
            ttc_fwd = ttc_left = ttc_right = 0.0
            headway_sec = 0.0
            vx = float(core.STATE.ego.velocity.x)
            back_min = rear_left_min = rear_right_min = 0.0
            rear_danger = 0.0
            safe_gate = 0.0
            spped_bonus = 0.0 
            accel_rear_relief_bonus = 0.0
            rear_close_pen = 0.0 
            rear_approach_pen = 0.0
            crash_type = "rear" if collided_from_rear else "front/side"
        else:
            # --- sensor slices & cones ---
            # observation[:16] are normalized distances in [0,1] (1.0 = far, 0.0 = very close)
            sensor_obs = observation[:16]

            # Forward & front-side cones (based on your sensor order)
            FWD_CONE    = (7, 9, 0, 10, 1)     # left_front, front_left_front, front, front_right_front, right_front
            LEFT_CONE   = (8, 7, 9)            # left_side_front, left_front, front_left_front
            RIGHT_CONE  = (10, 1, 11)          # front_right_front, right_front, right_side_front

            # Rear cones for rear-safety shaping
            BACK_CONE_NARROW = (3, 4, 5)       # right_back, back, left_back
            REAR_LEFT_CONE   = (15, 14, 5)     # left_side_back, back_left_back, left_back
            REAR_RIGHT_CONE  = (12, 13, 3)     # right_side_back, back_right_back, right_back

            # --- progress (distance) ---
            delta_dist = core.STATE.distance - prev_distance        # ~ ego.vx per tick
            fwd_min = float(np.min(sensor_obs[list(FWD_CONE)]))     # 0..1 (1 = far)
            # Harder gating: no tailgating reward; full reward when fwd_min >= ~0.65
            danger_gate = float(np.clip((fwd_min - 0.45) / 0.20, 0.0, 1.0))
            progress_r = 0.5 * max(delta_dist, 0.0) * danger_gate

            # --- Time-To-Collision (forward & front-sides), speed-aware ---
            MAX_SENSOR_PX = 1000.0
            FPS = 60.0
            vx = max(core.STATE.ego.velocity.x, 0.0)  # px/tick

            stuck_lane_change_bonus = 0.0
            # Condition: Are we stuck behind a car (high forward danger) and moving slowly?

            

            speed_bonus = 0.05 * float(np.clip((vx - 12.0) / 8.0, 0.0, 1.0))

            # forward TTC
            d_fwd_px = fwd_min * MAX_SENSOR_PX
            ttc_fwd = (d_fwd_px / max(vx, 1e-6)) / FPS
            TTC_CUTOFF_FWD = 1.2
            TTC_SCALE_FWD  = 4.5
            danger_fwd = float(np.clip((TTC_CUTOFF_FWD - ttc_fwd) / TTC_CUTOFF_FWD, 0.0, 1.0))
            pen_ttc_fwd = (danger_fwd ** 2) * TTC_SCALE_FWD

            # side TTCs (penalize turning into cars ahead-left / ahead-right)
            left_min  = float(np.min(sensor_obs[list(LEFT_CONE)]))
            right_min = float(np.min(sensor_obs[list(RIGHT_CONE)]))
            ttc_left  = (left_min * MAX_SENSOR_PX)  / max(vx, 1e-6) / FPS
            ttc_right = (right_min * MAX_SENSOR_PX) / max(vx, 1e-6) / FPS
            TTC_CUTOFF_SIDE = 1.2
            TTC_SCALE_SIDE  = 3.0
            danger_left  = float(np.clip((TTC_CUTOFF_SIDE - ttc_left) / TTC_CUTOFF_SIDE, 0.0, 1.0))
            danger_right = float(np.clip((TTC_CUTOFF_SIDE - ttc_right) / TTC_CUTOFF_SIDE, 0.0, 1.0))
            pen_ttc_side = (danger_left ** 2 + danger_right ** 2) * (TTC_SCALE_SIDE * 0.5)

            is_stuck = danger_fwd > 0.4 and vx < 10.0
            
            if is_stuck:
                # Is the left lane clear enough for a safe lane change?
                is_left_lane_clear = left_min > 0.8
                # Is the right lane clear enough?
                is_right_lane_clear = right_min > 0.8

                # If we are steering into a clear lane while stuck, give a bonus.
                if action_string == 'STEER_LEFT' and is_left_lane_clear:
                    stuck_lane_change_bonus = 0.2
                elif action_string == 'STEER_RIGHT' and is_right_lane_clear:
                    stuck_lane_change_bonus = 0.2

            # --- steer-into-danger penalty (front-sides) ---
            steer_into_danger = 0.0
            if action_string == 'STEER_LEFT' and left_min < (fwd_min + 0.10):
                steer_into_danger += 0.08 + 0.8 * max(0.0, 0.5 - left_min)
            if action_string == 'STEER_RIGHT' and right_min < (fwd_min + 0.10):
                steer_into_danger += 0.08 + 0.8 * max(0.0, 0.5 - right_min)

            # --- action-aware shaping (accel bad when danger; brake good when danger) ---
            act_shaping = 0.0
            if action_string == 'ACCELERATE' and danger_fwd > 0.35:
                act_shaping += 0.7 * (danger_fwd - 0.35)
            elif action_string == 'DECELERATE' and max(danger_fwd, danger_left, danger_right) > 0.35:
                act_shaping -= 0.35 * (max(danger_fwd, danger_left, danger_right) - 0.35)
            if action == "ACCELERATE" and ttc_fwd < 0.3:
            act_shaping += 1.0 * (0.3 - ttc_fwd)  # was 0.7 * (danger_fwd-0.35)

            # --- distance-keeping bonus (time headway) ---
            headway_sec = ttc_fwd
            HEADWAY_TARGET = 2.0
            headway_bonus = 0.08 * float(np.clip((headway_sec - HEADWAY_TARGET) / HEADWAY_TARGET, 0.0, 1.0))
            headway_bonus *= float(np.clip(vx / 12.0, 0.0, 1.0)) * danger_gate  # no idling exploit

            # --- small steering penalty to reduce jitter ---
            turn_pen = 0.02 if action_string in ('STEER_LEFT', 'STEER_RIGHT') else 0.0

            # --- Rear safety shaping (avoid being rear-ended) ---
            back_min       = float(np.min(sensor_obs[list(BACK_CONE_NARROW)]))   # 0..1
            rear_left_min  = float(np.min(sensor_obs[list(REAR_LEFT_CONE)]))
            rear_right_min = float(np.min(sensor_obs[list(REAR_RIGHT_CONE)]))

            rear_danger = float(np.clip((0.6 - back_min) / 0.6, 0.0, 1.0)) ** 2

            overcorrection_pen = 0.0
            # Condition: Are we accelerating while there's danger both behind AND in front?
            if action_string == 'ACCELERATE' and rear_danger > 0.3 and danger_fwd > 0.3:
                # The penalty is larger when both dangers are high, punishing the compounded risk.
                overcorrection_pen = 1.5 * rear_danger * danger_fwd
            
            # NEW: Penalty for cars getting too close behind
            # This is active even when not braking.
            rear_close_pen = 0.0
            if rear_danger > 0.3:
                rear_close_pen = 0.12 * rear_danger

            # NEW: Bonus for accelerating when danger is to the rear
            # This encourages the agent to speed up and create space.
            accel_rear_relief_bonus = 0.0
            if action_string == 'ACCELERATE' and rear_danger > 0.3 and danger_fwd < 0.2:
                accel_rear_relief_bonus = 0.15 * rear_danger

            # 1) Penalize braking when rear is dangerous and forward is relatively safe
            rear_brake_pen = 0.0
            if action_string == 'DECELERATE' and rear_danger > 0.25 and danger_fwd < 0.35:
                # MODIFICATION: Increased penalty from 0.2 to 0.4
                rear_brake_pen = 0.4 * rear_danger * (1.0 - danger_fwd)
            

            # 2) Penalize merging toward a rear-occupied side (avoid getting clipped)
            rear_merge_pen = 0.0
            if action_string == 'STEER_LEFT' and rear_left_min < 0.45:
                rear_merge_pen += 0.06 * (0.45 - rear_left_min) * float(np.clip(vx / 8.0, 0.0, 1.0))
            if action_string == 'STEER_RIGHT' and rear_right_min < 0.45:
                rear_merge_pen += 0.06 * (0.45 - rear_right_min) * float(np.clip(vx / 8.0, 0.0, 1.0))

            # --- compose reward ---
            survival = 0.01
            # MODIFICATION: Added new rewards and penalties
            reward = (
                progress_r
                - pen_ttc_fwd
                - pen_ttc_side
                - steer_into_danger
                - act_shaping
                - rear_brake_pen
                - rear_merge_pen
                - rear_close_pen
                - overcorrection_pen         # <-- ADDED PENALTY
                - turn_pen
                + headway_bonus
                + speed_bonus
                + accel_rear_relief_bonus
                + stuck_lane_change_bonus    # <-- ADDED BONUS
                + survival
            )
        # 7) Termination / truncation
        terminated = crashed
        truncated = core.STATE.ticks >= core.MAX_TICKS

        # 8) Info for diagnostics
        info = self._get_info()
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
            # rear diagnostics
            "back_min": float(back_min),
            "rear_left_min": float(rear_left_min),
            "rear_right_min": float(rear_right_min),
            "rear_danger": float(rear_danger),
            "crash_location": crash_location, # Add your new metric here
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