import mujoco
import mujoco.viewer
import time
import gymnasium as gym
import numpy as np
import json


def ik_dls_site(model, data, site_id, joint_ids, goal_pos, damping=0.01):
    """
    MuJoCo equivalent of PyBullet p.IK_DLS
    """
    # ------------------------------------------------
    # Current EE position (always read from data)
    # ------------------------------------------------
    ee_pos = data.site_xpos[site_id].copy()
    dx = goal_pos - ee_pos

    # ------------------------------------------------
    # Jacobian
    # ------------------------------------------------
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

    J = jacp[:, joint_ids]

    # ------------------------------------------------
    # Damped Least Squares
    # dq = Jᵀ (J Jᵀ + λ² I)⁻¹ dx
    # ------------------------------------------------
    JJt = J @ J.T
    dq = J.T @ np.linalg.solve(JJt + damping ** 2 * np.eye(3), dx)

    return dq

class SampleWS:
    
    def __init__(self, ws_path, x_range, y_range, z_range):
        self.ws_path = ws_path
        self.poses = self._read_json_ws()
        self.box_points = self._sample_box_points(x_range, y_range, z_range)
    
    def _read_json_ws(self):
        """
        read the workspace position and orientation information

        returns:
        pose_array: (n, 7) numpy array, 7 values: 3 position, 4 orientation values
        """
        with open(self.ws_path, 'r') as f:
            data = json.load(f)
        
        # init pos and orn list
        pose_list = []

        for i in range(len(data)):
            temp = data[i]
            pose_list.append(temp["ee_pos"] + temp["ee_orn"])
        
        return np.array(pose_list)
            
    
    def _sample_box_points(self, x_range, y_range, z_range):

        """
        sample a box of points from the workspace
        :param x_range: x lower and x upper
        :param y_range: y lower and y upper
        :param z_range: z lower and z upper
        """
        box_points = []

        for pose in self.poses:
            if pose[0] < x_range[1] and pose[0] > x_range[0]:
                if pose[1] < y_range[1] and pose[1] > y_range[0]:
                    if pose[2] < z_range[1] and pose[2] > z_range[0]:
                        box_points.append(pose)
        
        return np.array(box_points)
    

class RobotEnv(gym.Env):

    def __init__(self, 
                 xml_path: str, 
                 ws_path: str, 
                 ee_site: str, 
                 joints_names: list[str], 
                 render: bool):
        super().__init__()

        # ---------------------------
        # RL parameters
        # ---------------------------
        self.pos_dim       = 3     # position of the goal dimension: 3
        self.pos_threshold = 0.01  # position error threshold in meters
        self.step_count    = 0     # step count number
        self.max_steps     = 200   # maximum step per episode
        self.action_scale  = 0.005  # action scale per step for the end-effector

        # set the observation space and action space
        # observation is of 8 values:
        # 3 for dx dy dz: end-effector distance to the goal
        # 3 for vx vy vz: end-effector velocity
        # 1 for gripper state: open or closed: 0 is open, -0.039 is close
        # 1 for on goal state: if the end-effector hold there for a significant amount of time
        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(low=-10, high=10, shape=(8,), dtype=np.float32),
            "achieved_goal": gym.spaces.Box(low=-10, high=10, shape=(self.pos_dim,), dtype=np.float32),
            "desired_goal": gym.spaces.Box(low=-10, high=10, shape=(self.pos_dim,), dtype=np.float32),
        })
        
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.pos_dim,), dtype=np.float32)
        
        # ---------------------------
        # Mujoco setup
        # ---------------------------
        self.model  = mujoco.MjModel.from_xml_path(xml_path)
        self.data   = mujoco.MjData(self.model)
        self.viewer = None
        if render:
            self.render()

        # ---------------------------
        # Robot joints
        # ---------------------------
        self.init_q        = np.array([0, 0.57, -0.73, -1.24, 0, -0.77, 0.0, 0.0])
        self.joints_names  = joints_names
        self.actuator_ids  = [self.model.actuator(name).id for name in self.joints_names]
        self.joint_dof_ids = [0, 1, 2, 3, 4, 5, 6, 7]

        # ---------------------------
        # End-effector
        # ---------------------------
        self.ee_site_id = self.model.site(ee_site).id

        # ---------------------------
        # Workspace goals
        # ---------------------------
        self.task_space_XYZ = ((-0.4, 0.4), (0.1, 0.4), (0.1, 0.3))
        self.ws_path        = ws_path
        self.goal_poses     = self._read_json_ws()
        self.goal_pose      = None
        self.goal_mocap_id  = None
    
    def render(self):
        if not hasattr(self, "viewer") or self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.azimuth   = 225
            self.viewer.cam.elevation = -25
            self.viewer.cam.distance  = 2.0
            self.viewer.cam.lookat[:] = [0, 0, 0]

        self.viewer.sync()
    
    def _get_ee_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the current end-effector state: position and velocity """

        # ---------------------------------------
        # Get the current end-effector position
        # ---------------------------------------
        ee_pos = self.data.site(self.ee_site_id).xpos

        # ------------------------------------------------
        # Compute the jacobian to get the linear velocity
        # ------------------------------------------------

        # Step 1: allocate memory (zeros are just initialization)
        jacp   = np.zeros((3, self.model.nv))
        jacr   = np.zeros((3, self.model.nv))

        # Step 2: MuJoCo fills these arrays
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)

        # Step 3: Compute the linear velocity: x_dot = jacobian_v * q_dot
        ee_vel = jacp @ self.data.qvel

        return ee_pos, ee_vel
    
    def _read_json_ws(self):
        """
        read the workspace position and orientation information

        returns:
        pose_array: (n, 7) numpy array, 7 values: 3 position, 4 orientation values
        """
        x, y, z = self.task_space_XYZ
        sample_ws = SampleWS(self.ws_path, x, y, z)
        return sample_ws.box_points
    
    def visualize_sample_ws(self):
        for pose in self.goal_poses:
            self.data.mocap_pos[self.goal_mocap_id][:] = pose[0:3]
            mujoco.mj_forward(self.model, self.data)

            if self.viewer:
                self.viewer.sync()
            time.sleep(0.001)
                
    def _return_to_init_joints(self):
        """  
        return to initial joint values of the robot
        """
        
        # --------------------------------
        # Set joint positions directly
        # --------------------------------
        self.data.qpos[:len(self.init_q)] = self.init_q
        self.data.qvel[:] = 0.0
        
        # --------------------------------------------------------
        # Set the actuator controls to match initial joints
        # --------------------------------------------------------
        for i, act_id in enumerate(self.actuator_ids):
            self.data.ctrl[act_id] = self.init_q[i]
        
        # --------------------------------------------------------
        # Forward simulate once to update positions
        # --------------------------------------------------------
        mujoco.mj_forward(self.model, self.data)
    
    def _rand_goal_pose(self) -> np.ndarray:
        return self.goal_poses[np.random.choice(self.goal_poses.shape[0])]
    
    def _set_goal_marker(self):
        """ 
        Set the goal marker in the goal position
        """
        if self.goal_mocap_id is None:
            goal_marker_id     = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_marker")
            self.goal_mocap_id = self.model.body_mocapid[goal_marker_id]

        self.data.mocap_pos[self.goal_mocap_id][:] = self.goal_pose[0:3]
        mujoco.mj_forward(self.model, self.data)

    def reset(self, seed=None):
        """
        Reset the robot state: all joint states, the target state
        """
        super().reset(seed=seed)

        # ---------------------------
        # Initialization of resetting
        # ---------------------------
        self.step_count    = 0                      # init step count
        self.goal_pose     = self._rand_goal_pose() # randomize the goal pose
        
        # -----------------------------------
        # Reset robot joints and goal marker
        # -----------------------------------
        mujoco.mj_resetData(self.model, self.data)
        self._return_to_init_joints()
        self._set_goal_marker()
        
        # ---------------------------------------------------------------------------------
        # Form the observation: dx, dy, dz, vx, vy, vz, gripper state, reach on goal state
        # ---------------------------------------------------------------------------------
        ee_pos, ee_vel = self._get_ee_state()

        obs = np.concatenate([
            ee_pos - self.goal_pose[0:3],
            ee_vel,
            np.array([0.0]),
            np.array([0.0])
        ])
        
        return {
            "observation": obs.astype(np.float32),
            "achieved_goal": ee_pos.astype(np.float32),
            "desired_goal": self.goal_pose[0:3].astype(np.float32),
        }, {}
    
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, ee_vel, terminated: bool):
        """
        Compute the reward based on the achieved goal and desired goal
        
        :param achieved_goal: current end-effector goal position
        :type achieved_goal: np.ndarray
        :param desired_goal: desired end-effector goal position
        :type desired_goal: np.ndarray
        :param terminated: whether the robot should stop, if terminated is true, then the robot reaches the goal
        :type terminated: bool
        """
        # --------------------------------------------------------
        # [1] Distance-based reward
        # --------------------------------------------------------
        dist = np.linalg.norm(achieved_goal - desired_goal)
        r    = -dist
        r    -= 0.1 * np.linalg.norm(ee_vel)**2  # penalize speed
        
        # --------------------------------------------------------
        # [2] Success Bonus
        # --------------------------------------------------------
        success = dist < self.pos_threshold
        
        if success:
            terminated = True
            r += 20
        
        # --------------------------------------------------------
        # [3] Collision penalty with floor
        # --------------------------------------------------------
        floor_geom_id = self.model.geom("ground").id
        collision     = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 == floor_geom_id or contact.geom2 == floor_geom_id:
                collision = True
                break

        if collision:
            r -= 10.0           # large penalty
            terminated = True   # End episode on collision

        return r, terminated, [1. if success else 0.]

    def step(self, action):
        """  
        step the action and return observation
        """
        
        # --------------
        # Bookkeeping
        # --------------
        terminated      = False  # set the terminated to False
        self.step_count += 1     # step count plus 
        
        # ------------------------------
        # Scale and clip the action
        # ------------------------------
        action = np.clip(action, -1, 1) * self.action_scale
        
        # ------------------------------------------------------
        # Get the current end effector reference point position
        # ------------------------------------------------------
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        
        # --------------------------
        # Set the target position
        # --------------------------
        goal_pos = ee_pos + action
        
        # ---------------------------
        # Stepping the robot joints
        # ---------------------------
        dq          = ik_dls_site(self.model, self.data, self.ee_site_id, self.joint_dof_ids, goal_pos, 0.01)
        qpos        = self.data.qpos[self.joint_dof_ids]
        qpos_target = qpos + dq

        # Apply to actuators
        self.data.ctrl[self.actuator_ids[0:6]] = qpos_target[0:6]  # apply to arm actuators
        self.data.ctrl[self.actuator_ids[6]]   = 0.0
        self.data.ctrl[self.actuator_ids[7]]   = 0.0

        # Step the simulation for a few substeps to make the robot move
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
            if self.viewer:
                self.viewer.sync()
        
        # ----------------------------------------------------------------------------------
        # Get the current end effector reference point position and velocity after stepping
        # ----------------------------------------------------------------------------------
        ee_pos, ee_vel = self._get_ee_state()

        # Form the reward, terminated, and truncated
        truncated = self.step_count >= self.max_steps
        reward, terminated, on_goal_flag = self.compute_reward(ee_pos, self.goal_pose[0:3], ee_vel, terminated)

        obs = np.concatenate([
            ee_pos - self.goal_pose[0:3],
            ee_vel,
            np.array([0.0]),
            on_goal_flag,
        ])

        if self.viewer:
            self.viewer.sync()

        return {
            "observation": obs.astype(np.float32),
            "achieved_goal": ee_pos.astype(np.float32),
            "desired_goal": self.goal_pose[0:3].astype(np.float32),
        }, reward, terminated, truncated, {}

    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    xml_path     = "robot.xml"
    ws_path      = "robot_workspace.json"
    joints_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper_left_finger_joint", "gripper_right_finger_joint"]
    
    env = RobotEnv(xml_path=xml_path, ws_path=ws_path, joints_names=joints_names, ee_site="ee_site", render=True)

    obs, _ = env.reset()
    
    print("===================================================================")
    print(f"Goal poses shape:  {env.goal_poses.shape}")
    print(f"Joint actuators:   {env.actuator_ids}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}")
    print(f"Current goal pose: position: {env.goal_pose[0:3]}, orientation: {env.goal_pose[3:7]}")
    print("===================================================================")
    
    # ---------------------------
    # Manual rollout loop
    # ---------------------------
    done = False

    while not done:

        # Random action (for testing)
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        ee_pos = obs["achieved_goal"]
        dist = np.linalg.norm(ee_pos - obs["desired_goal"])

        print(
            f"Step {env.step_count:3d} | "
            f"Action: {action.round(3)} | "
            f"Dist: {dist:.3f} | "
            f"Reward: {reward}"
        )

        time.sleep(0.1)  # slow down visualization

    print("Episode finished.")
    env.close()

