import numpy as np

class Preprocessor():

    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray):
        q_w = q[-1]
        q_vec = q[:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * (q_w * 2.0)
        c = q_vec * (np.dot(q_vec, v) * 2.0)  
        return a - b + c 

    def modify_state(self, obs, info):
        
        quat = info["robot_quat"]
        base_ang_vel = info["robot_gyro"]

        project_gravity = self.quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        obs = np.hstack((obs,
                         project_gravity,
                         base_ang_vel))

        return obs
