import numpy as np
import pkg_resources
import rowan
import json


DEFAULT_CONTROL_PARAM_FILE = pkg_resources.resource_filename(__name__, '../params/controller.json')
DEFAULT_PX4_PARAM_FILE = pkg_resources.resource_filename(__name__, '../params/px4.json')
DEFAULT_QUAD_PARAMETER_FILE = pkg_resources.resource_filename(__name__, '../params/quadrotor.json')


def readparamfile(filename, params=None):
    if params is None:
        params = {}
    with open(filename) as file:
        params.update(json.load(file))
    return params


class PIDController():
    def __init__(self,
                 quadparamfile=DEFAULT_QUAD_PARAMETER_FILE,
                 ctrlparamfile=DEFAULT_CONTROL_PARAM_FILE,
                 px4paramfile=DEFAULT_PX4_PARAM_FILE,
                 pid_params=None):
        # PX4 params
        self.px4_params = readparamfile(px4paramfile)
        self.px4_params['angrate_max'] = np.array((self.px4_params['MC_ROLLRATE_MAX'],
                                                  self.px4_params['MC_PITCHRATE_MAX'],
                                                  self.px4_params['MC_YAWRATE_MAX']))
        self.px4_params['attitude_gain_P'] = np.diag((self.px4_params['MC_ROLL_P'],
                                                  self.px4_params['MC_PITCH_P'],
                                                  self.px4_params['MC_YAW_P']))
        self.px4_params['angacc_max'] = np.array(self.px4_params['angacc_max'])
        
        # Quadrotor params
        self.params = readparamfile(quadparamfile)

        # Controller params
        self.params = readparamfile(filename=ctrlparamfile, params=self.params)

        # PID params
        if pid_params is None:
            pid_params = readparamfile('params/pid.json')
        self.params['Lam_xy'] = pid_params['Lam_xy']
        self.params['K_xy'] = pid_params['K_xy']
        self.params['Lam_z'] = pid_params['Lam_z']
        self.params['K_z'] = pid_params['K_z']
        self.params['i'] = pid_params['i']
        self.calculate_gains()

    def calculate_gains(self):
        self.params['K_p'] = np.diag([
            self.params['Lam_xy'] * self.params['K_xy'],
            self.params['Lam_xy'] * self.params['K_xy'],
            self.params['Lam_z'] * self.params['K_z']
        ])
        self.params['K_i'] = np.diag(
            [self.params['i'], self.params['i'], self.params['i']
        ])
        self.params['K_d'] = np.diag(
            [self.params['K_xy'], self.params['K_xy'], self.params['K_z']
        ])
        self.B = np.array([
            self.params['C_T'] * np.ones(4), self.params['C_T'] *
            self.params['l_arm'] * np.array([-1., -1., 1., 1.]),
            self.params['C_T'] * self.params['l_arm'] *
            np.array([-1., 1., 1., -1.]),
            self.params['C_q'] * np.array([-1., 1., -1., 1.])
        ])

    def print_pid(self):
        print("K_p", self.params['K_p'])
        print("K_i", self.params['K_i'])
        print("K_d", self.params['K_d'])

    def reset_base_controller(self):
        self.F_r_dot = None
        self.F_r_last = None
        self.t_last = None
        self.t_last_wind_update = -self.params['wind_update_period']
        self.int_error = np.zeros(3)
        self.dt = 0.
        self.motor_speed = np.zeros(4)
    
    def reset_controller(self):
        pass

    def get_q(self, F_r, yaw=0., max_angle=np.pi):
        q_world_to_yaw = rowan.from_euler(0., 0., yaw, 'xyz')
        rotation_axis = np.cross((0, 0, 1), F_r)
        if np.allclose(rotation_axis, (0., 0., 0.)):
            unit_rotation_axis = np.array((1., 0., 0.,))
        else:
            unit_rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            rotation_axis /= np.linalg.norm(F_r)
        rotation_angle = np.arcsin(np.linalg.norm(rotation_axis))
        if F_r[2] < 0:
            rotation_angle = np.pi - rotation_angle
        if rotation_angle > max_angle:
            rotation_angle = max_angle
        q_yaw_to_body = rowan.from_axis_angle(unit_rotation_axis,
                                              rotation_angle)
        q_r = rowan.multiply(q_world_to_yaw, q_yaw_to_body)
        return rowan.normalize(q_r)

    def get_Fr(self, X, imu, pd, vd, ad, meta_adapt_trigger):
        p_error = X[0:3] - pd
        v_error = X[7:10] - vd
        self.int_error += self.dt * p_error
        a_r = - self.params['K_p'] @ p_error - self.params['K_d'] @ v_error - \
                self.params['K_i'] @ self.int_error + ad
        F_r = (a_r * self.params['m']) + np.array(
            [0., 0., self.params['m'] * self.params['g']])

        if self.F_r_last is None:
            self.F_r_dot = np.zeros(3)
        else:
            lam = np.exp(-self.dt / self.params['force_filter_time_const'])
            self.F_r_dot *= lam
            self.F_r_dot += (1 - lam) * (F_r - self.F_r_last) / self.dt
        self.F_r_last = F_r.copy()
        return F_r, self.F_r_dot

    def position(self, X, imu, pd, vd, ad, last_wind_update, t):
        if self.t_last is None:
            self.t_last = t
        else:
            self.dt = t - self.t_last
        if (self.t_last_wind_update < last_wind_update):
            self.t_last_wind_update = last_wind_update
            meta_adapt_trigger = True
        else:
            meta_adapt_trigger = False

        yaw = 0.
        self.t_last = t
        F_r, F_r_dot = self.get_Fr(X, imu=imu, pd=pd, vd=vd, ad=ad,
                                   meta_adapt_trigger=meta_adapt_trigger)
        T_r_prime = np.linalg.norm(F_r + self.params['thrust_delay'] * F_r_dot)
        q_r_prime = self.get_q(F_r + self.params['attitude_delay'] * F_r_dot,
                               yaw)
        F_r_prime = rowan.to_matrix(q_r_prime) @ np.array((0, 0, T_r_prime))

        T_r_prime = np.linalg.norm(F_r_prime)
        q_r_prime = self.get_q(F_r_prime, yaw)
        return T_r_prime, q_r_prime

    def attitude(self, q, q_sp):
        q_error = rowan.multiply(rowan.inverse(q), q_sp)
        omega_sp = 2 * self.px4_params['attitude_gain_P'] @ (
            np.sign(q_error[0]) * q_error[1:])
        self.limit(omega_sp, self.px4_params['angrate_max'])
        return omega_sp
    
    def limit(self, array, upper_limit, lower_limit=None):
        if lower_limit is None:
            lower_limit = - upper_limit
        array[array > upper_limit] = upper_limit[array > upper_limit]
        array[array < lower_limit] = lower_limit[array < lower_limit]
    
    def get_u(self, T_sp, w_sp, w):
        w_error = w_sp - w
        alpha_sp = np.diag([1,1,1]) @ w_error + np.cross(w.T, np.dot(self.params['J'], w).T).T
        self.limit(alpha_sp, self.px4_params['angacc_max'])
        omega_squared = np.linalg.solve(self.B, np.concatenate(((T_sp,), alpha_sp)))
        omega = np.sqrt(np.maximum(omega_squared, self.params['motor_min_speed']))
        omega = np.minimum(omega, self.params['motor_max_speed'])
        self.motor_speed = omega
    
    def get_action(self, obs, t, pd, vd, ad, imu, t_last_wind_update):
        X = obs[0:13]
        X[0:3] += pd
        X[7:10] += vd
        # 位置控制器：获取期望总推力 T_sp 和期望姿态四元数 q_sp
        T_sp, q_sp = self.position(X=X, imu=imu, pd=pd, vd=vd, ad=ad, t=t, last_wind_update=t_last_wind_update)
        # 姿态控制器：获取期望角速度 w_sp
        w_sp = self.attitude(q=X[3:7], q_sp=q_sp)
        # ctbr控制命令
        action = np.array([T_sp / self.params['m'], *w_sp])
        # 记录该命令得到的电机转速
        self.get_u(T_sp=T_sp, w_sp=w_sp, w=X[10:])
        return action