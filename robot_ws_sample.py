import numpy as np
import pybullet as p
import pybullet_data
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import PolyCollection
import json


class ComputeWS:

    def __init__(self, urdf_path, num_samples, ee_link_name="ee_ref_link"):
        
        # init physics client
        p.connect(p.DIRECT)
        # p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0,0,0])
        plane_id = p.loadURDF("plane.urdf")
        
        # set the urdf path and read it in pybullet
        self.urdf_path   = urdf_path
        self.robot_id    = self._read_urdf()

        # get the active joint indices and its joint limits
        self.active_joint_indices, self.active_joint_limits = self._access_active_joints()

        # set the number of samples
        self.num_samples = num_samples

        # sample all q 
        self.q_values = self._sampling_joints()

        # set the name of the end-effector reference point
        self.ee_link     = ee_link_name

        # set the voxel size for plotting 
        self.voxel_size  = 0.05
        
        # compute the workspace positions and orientations
        self.ws_positions, self.ws_orientations = self._compute_ws()

        # print the ws range information
        self._print_ws_range()

        # plot the workspace
        # self._plot_workspace(self.ws_positions)
    
    def _print_ws_range(self):
        """  
        print the workspace range for both positions and orientations
        """

        pos_min = np.min(self.ws_positions, axis=0)
        pos_max = np.max(self.ws_positions, axis=0)
        print("Position range:")
        print("X: [{:.3f}, {:.3f}], Y: [{:.3f}, {:.3f}], Z: [{:.3f}, {:.3f}]".format(
            pos_min[0], pos_max[0], pos_min[1], pos_max[1], pos_min[2], pos_max[2]))
        
        orn_min = np.min(self.ws_orientations, axis=0)
        orn_max = np.max(self.ws_orientations, axis=0)
        print("\nOrientation range (radians):")
        print("Roll: [{:.3f}, {:.3f}], Pitch: [{:.3f}, {:.3f}], Yaw: [{:.3f}, {:.3f}]".format(
            orn_min[0], orn_max[0], orn_min[1], orn_max[1], orn_min[2], orn_max[2]))
    
    def _read_urdf(self):
        robot_id = p.loadURDF(self.urdf_path)
        return robot_id
    
    def _access_active_joints(self):
        """
        access the joint information
        """

        # init the joints dict
        joint_indices = []
        joint_limits  = []
        
        num_joints = p.getNumJoints(self.robot_id)

        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_type = info[2]

            if joint_type in [p.JOINT_REVOLUTE]:
                joint_indices.append(i)
                joint_limits.append([info[8], info[9]])
        
        return joint_indices, joint_limits
    
    def _get_ee_link_index(self):
        num_joints = p.getNumJoints(self.robot_id)

        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            link_name = info[12].decode('utf-8')

            if link_name == self.ee_link:
                return i
    
    def _sampling_joints(self):
        """
        Sample all joint values within its limit
        """
        
        # init joint values 
        q_values = []
        
        # start to sample for each joint index
        for _ in range(self.num_samples):
            q_value = []
            for i in range(len(self.active_joint_indices)):
                q_value.append(np.random.uniform(self.active_joint_limits[i][0], self.active_joint_limits[i][1]))
            q_values.append(q_value)
        
        return np.array(q_values)
    
    def _compute_ws(self):
        """
        Compute workspace positions and orientations
        """

        ws_positions = []
        ws_orientations = []
        ee_link_index = self._get_ee_link_index()

        for q in self.q_values:
            
            # set joint positions
            for idx, val in zip(self.active_joint_indices, q):
                p.resetJointState(self.robot_id, idx, val)
            
            # get the end-effector world pose
            state = p.getLinkState(self.robot_id, ee_link_index)
            pos   = state[4]
            orientation = p.getEulerFromQuaternion(state[5])
            
            # optionally see the pose in pybullet
            # p.addUserDebugLine(pos, tuple(np.array(pos) + np.array([0,0,0.01])), [1,0,0], lineWidth=5, lifeTime=0.5)
            # time.sleep(1)

            ws_positions.append(pos)
            ws_orientations.append(orientation)
        
        return np.array(ws_positions), np.array(ws_orientations)
    
    def save_ws(self):

        # initialization
        data = []
        ee_link_index = self._get_ee_link_index()

        # loop all q in the q values
        for q in self.q_values:

            # set the joint positions
            for idx, val in zip(self.active_joint_indices, q):
                p.resetJointState(self.robot_id, idx, val)
            
            # get the end-effector world pose
            state = p.getLinkState(self.robot_id, ee_link_index)
            pos = state[4]
            orn = state[5]

            # save the pos and ori
            data.append({
                "q": q.tolist(),
                "ee_pos": pos,
                "ee_orn": orn,
            })
        
        # write to json
        file_str = "env/robot_workspace.json"
        with open(file_str, 'w') as f:
            json.dump(data, f, indent=4)
        
        # print the msg
        print(f"File of the workspace saved at the env directory with file name: {file_str}")



    def _voxelization(self, point_cloud: np.ndarray):
        """
        voxelize the point cloud
        """
        min_coords = np.min(point_cloud, axis=0)

        # compute the voxel indices
        voxel_indices = np.floor((point_cloud - min_coords) / self.voxel_size).astype(int)

        return np.unique(voxel_indices, axis=0)
    
    def _compute_voxel_faces(self, voxel_pos):

        """
        Compute the voxel face coordinates
        input:
        voxel_pos: one voxel position

        output: 
        faces of the voxel

        """
        x, y, z = voxel_pos

        # 6 faces
        bottom_face = [(x, y, z), 
                       (x+self.voxel_size, y, z), 
                       (x+self.voxel_size, y+self.voxel_size, z), 
                       (x, y+self.voxel_size, z)]
        
        top_face    = [(x, y, z+self.voxel_size), 
                       (x+self.voxel_size, y, z+self.voxel_size), 
                       (x+self.voxel_size, y+self.voxel_size, z+self.voxel_size), 
                       (x, y+self.voxel_size, z+self.voxel_size)]
        
        front_face  = [(x, y, z), 
                       (x, y, z+self.voxel_size), 
                       (x, y+self.voxel_size, z+self.voxel_size), 
                       (x, y+self.voxel_size, z)]
        
        back_face   = [(x+self.voxel_size, y, z), 
                       (x+self.voxel_size, y, z+self.voxel_size), 
                       (x+self.voxel_size, y+self.voxel_size, z+self.voxel_size), 
                       (x+self.voxel_size, y+self.voxel_size, z)]
        
        left_face   = [(x, y, z), 
                       (x, y, z+self.voxel_size), 
                       (x+self.voxel_size, y, z+self.voxel_size), 
                       (x+self.voxel_size, y, z)]
        
        right_face  = [(x, y+self.voxel_size, z), 
                       (x, y+self.voxel_size, z+self.voxel_size), 
                       (x+self.voxel_size, y+self.voxel_size, z+self.voxel_size), 
                       (x+self.voxel_size, y+self.voxel_size, z)]
        
        faces = [bottom_face, top_face, front_face, back_face, left_face, right_face]

        return faces
    
    def _plot_workspace(self, point_cloud):
        
        """ 
        plot the workspace 
        """
        
        # figure initialization
        fig_original, ax_original = plt.subplots(
            subplot_kw={'projection': '3d'}
        )

        ax_original.set_xlabel('X / m', fontsize=15)  # set x label
        ax_original.set_ylabel('Y / m', fontsize=15)  # set y label
        ax_original.set_zlabel('Z / m', fontsize=15)  # set z label
        
        # point cloud computation of positions
        min_coords      = np.min(point_cloud, axis=0)
        occupied_voxels = self._voxelization(point_cloud)
        voxel_positions = np.array(list(occupied_voxels)) * self.voxel_size + min_coords
        
        # set the view of plot
        max_coords = np.max(point_cloud, axis=0)
        center = (min_coords + max_coords) / 2
        radius = np.max(max_coords - min_coords) / 2

        ax_original.set_xlim(center[0] - radius, center[0] + radius)
        ax_original.set_ylim(center[1] - radius, center[1] + radius)
        ax_original.set_zlim(center[2] - radius, center[2] + radius)

        ax_original.quiver(0,0,0, 0.2,0,0, color='r')
        ax_original.quiver(0,0,0, 0,0.2,0, color='g')
        ax_original.quiver(0,0,0, 0,0,0.2, color='b')

        # plot the voxels
        for voxel_pos in voxel_positions:
            voxel_faces = self._compute_voxel_faces(voxel_pos)
            
            # add the voxel with face coloring
            ax_original.add_collection3d(Poly3DCollection(voxel_faces, facecolors='cyan', edgecolors='k', linewidths=0.2, alpha=0.8))

        # --- Prepare faces for projection views ---
        top_faces = []
        front_faces = []
        left_faces = []

        for voxel_pos in voxel_positions:
            voxel_faces = self._compute_voxel_faces(voxel_pos)

            for face in voxel_faces:
                # Top view: project onto XY
                top_faces.append([(x, y) for x, y, _ in face])
                # Front view: project onto XZ
                front_faces.append([(x, z) for x, _, z in face])
                # Left view: project onto YZ
                left_faces.append([(y, z) for _, y, z in face])

        # --- Create Top view (XY plane) ---
        fig_top, ax_top = plt.subplots(figsize=(6, 5))
        top_collection = PolyCollection(top_faces, facecolors='cyan', edgecolors='k', linewidths=0.2, alpha=0.8)
        ax_top.add_collection(top_collection)
        ax_top.set_title("Top View (XY plane)", fontsize=14)
        ax_top.set_xlabel("X / m", fontsize=12)
        ax_top.set_ylabel("Y / m", fontsize=12)
        ax_top.autoscale_view()
        ax_top.set_aspect('equal')
        ax_top.grid(True, linestyle='--', linewidth=0.3)

        # --- Create Front view (XZ plane) ---
        fig_front, ax_front = plt.subplots(figsize=(6, 5))
        front_collection = PolyCollection(front_faces, facecolors='cyan', edgecolors='k', linewidths=0.2, alpha=0.8)
        ax_front.add_collection(front_collection)
        ax_front.set_title("Front View (XZ plane)", fontsize=14)
        ax_front.set_xlabel("X / m", fontsize=12)
        ax_front.set_ylabel("Z / m", fontsize=12)
        ax_front.autoscale_view()
        ax_front.set_aspect('equal')
        ax_front.grid(True, linestyle='--', linewidth=0.3)

        # --- Create Left view (YZ plane) ---
        fig_left, ax_left = plt.subplots(figsize=(6, 5))
        left_collection = PolyCollection(left_faces, facecolors='cyan', edgecolors='k', linewidths=0.2, alpha=0.8)
        ax_left.add_collection(left_collection)
        ax_left.set_title("Left View (YZ plane)", fontsize=14)
        ax_left.set_xlabel("Y / m", fontsize=12)
        ax_left.set_ylabel("Z / m", fontsize=12)
        ax_left.autoscale_view()
        ax_left.set_aspect('equal')
        ax_left.grid(True, linestyle='--', linewidth=0.3)

        plt.show()


def main():

    # load the urdf from file
    urdf_path = 'env/robot.urdf'
    num_samples = 150000
    ws_obj = ComputeWS(urdf_path, num_samples)
    ws_obj.save_ws()


if __name__ == "__main__":
    main()