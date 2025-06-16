import numpy as np
import pkg_resources
import rowan
import json


DEFAULT_CONTROL_PARAM_FILE = pkg_resources.resource_filename(__name__, 'params/controller.json')
DEFAULT_PX4_PARAM_FILE = pkg_resources.resource_filename(__name__, 'params/px4.json')
DEFAULT_QUAD_PARAMETER_FILE = pkg_resources.resource_filename(__name__, 'params/quadrotor.json')

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
                 given_pid=False,
                 Lam_xy=0,
                 K_xy=0,
                 Lam_z=0,
                 K_z=0,
                 i=0):
        # PX4 params
        self.px4_params = readparamfile(px4paramfile)
        self.px4_params['angrate_max'] = np.array((self.px4_params['MC_ROLLRATE_MAX'],
                                                  self.px4_params['MC_PITCHRATE_MAX'],
                                                  self.px4_params['MC_YAWRATE_MAX']))
        self.px4_params['angrate_gain_P'] = np.diag((self.px4_params['MC_ROLLRATE_P'],
                                                  self.px4_params['MC_PITCHRATE_P'],
                                                  self.px4_params['MC_YAWRATE_P']))
        self.px4_params['angrate_gain_I'] = np.diag((self.px4_params['MC_ROLLRATE_I'],
                                                  self.px4_params['MC_PITCHRATE_I'],
                                                  self.px4_params['MC_YAWRATE_I']))
        self.px4_params['angrate_gain_D'] = np.diag((self.px4_params['MC_ROLLRATE_D'],
                                                  self.px4_params['MC_PITCHRATE_D'],
                                                  self.px4_params['MC_YAWRATE_D']))
        self.px4_params['angrate_gain_K'] = np.diag((self.px4_params['MC_ROLLRATE_K'],
                                                  self.px4_params['MC_PITCHRATE_K'],
                                                  self.px4_params['MC_YAWRATE_K']))
        self.px4_params['angrate_int_lim'] = np.array((self.px4_params['MC_RR_INT_LIM'],
                                                   self.px4_params['MC_PR_INT_LIM'],
                                                   self.px4_params['MC_YR_INT_LIM']))
        self.px4_params['attitude_gain_P'] = np.diag((self.px4_params['MC_ROLL_P'],
                                                  self.px4_params['MC_PITCH_P'],
                                                  self.px4_params['MC_YAW_P']))
        self.px4_params['angacc_max'] = np.array(self.px4_params['angacc_max'])
        self.px4_params['J'] = np.array(self.px4_params['J'])
        self.B = None

        # Quadrotor params
        self.params = readparamfile(quadparamfile)

        # Controller params
        self.params = readparamfile(filename=ctrlparamfile, params=self.params)

        # PID params
        self.given_pid = given_pid
        if (given_pid):
            self.params['Lam_xy'] = Lam_xy
            self.params['K_xy'] = K_xy
            self.params['Lam_z'] = Lam_z
            self.params['K_z'] = K_z
            self.params['K_i'] = np.diag([i, i, i])

    def calculate_gains(self):
        self.params['K_i'] = np.array(self.params['K_i'])
        self.params['K_p'] = np.diag([
            self.params['Lam_xy'] * self.params['K_xy'],
            self.params['Lam_xy'] * self.params['K_xy'],
            self.params['Lam_z'] * self.params['K_z']
        ])
        self.params['K_d'] = np.diag(
            [self.params['K_xy'], self.params['K_xy'], self.params['K_z']])
        self.B = np.array([
            self.params['C_T'] * np.ones(4), self.params['C_T'] *
            self.params['l_arm'] * np.array([-1., -1., 1., 1.]),
            self.params['C_T'] * self.params['l_arm'] *
            np.array([-1., 1., 1., -1.]),
            self.params['C_q'] * np.array([-1., 1., -1., 1.])
        ])
        # print("K_p", self.params['K_p'])
        # print("K_i", self.params['K_i'])
        # print("K_d", self.params['K_d'])

    def reset_controller(self):
        self.w_error_int = np.zeros(3)
        self.w_filtered = np.zeros(3)
        self.w_filtered_last = np.zeros(3)

        self.calculate_gains()
        self.F_r_dot = None
        self.F_r_last = None
        self.t_last = None
        self.t_last_wind_update = -self.params['wind_update_period']
        self.p_error = np.zeros(3)
        self.v_error = np.zeros(3)
        self.int_error = np.zeros(3)
        self.dt = 0.
        self.dt_inv = 0.

    def reset_time(self):
        self.t_posctrl = -0.0
        self.t_attctrl = -0.0
        self.t_angratectrl = -0.0

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

    def get_Fr(self, X, pd, vd, ad):
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

    def position(self, X, pd, vd, ad, last_wind_update, t):
        if self.t_last is None:
            self.t_last = t
        else:
            self.dt = t - self.t_last
        if (self.t_last_wind_update < last_wind_update):
            self.t_last_wind_update = last_wind_update

        yaw = 0.
        self.t_last = t
        F_r, F_r_dot = self.get_Fr(X, pd=pd, vd=vd, ad=ad)
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

    def angrate(self, w, w_sp, dt):
        w_error = w_sp - w
        self.w_error_int += dt * w_error
        self.limit(self.w_error_int, self.px4_params['angrate_int_lim'])

        const_w_filter = np.exp(-dt / self.px4_params['w_filter_time_const'])
        self.w_filtered *= const_w_filter
        self.w_filtered += (1 - const_w_filter) * w

        w_filtered_derivative = (self.w_filtered - self.w_filtered_last) / dt
        self.w_filtered_last[:] = self.w_filtered[:]  # Python is a garbage language

        alpha_sp = self.px4_params['angrate_gain_K'] \
                    @ (self.px4_params['angrate_gain_P'] @ w_error
                       + self.px4_params['angrate_gain_I'] @ self.w_error_int
                       - self.px4_params['angrate_gain_D'] @ w_filtered_derivative)
        self.limit(alpha_sp, self.px4_params['angacc_max'])
        return alpha_sp

    def limit(self, array, upper_limit, lower_limit=None):
        if lower_limit is None:
            lower_limit = - upper_limit
        array[array > upper_limit] = upper_limit[array > upper_limit]
        array[array < lower_limit] = lower_limit[array < lower_limit]

    def mixer(self, torque_sp, T_sp):
        omega_squared = np.linalg.solve(self.B, np.concatenate(((T_sp,), torque_sp)))
        omega = np.sqrt(np.maximum(omega_squared, self.params['motor_min_speed']))
        omega = np.minimum(omega, self.params['motor_max_speed'])
        return omega
    
    def getu(self, X, t, pd, vd, ad, imu, t_last_wind_update):
        if t >= self.t_posctrl:      # 位置控制器：获取期望总推力 T_sp 和期望姿态四元数 q_sp
            # pd, vd, ad = traj(t)
            T_sp, q_sp = self.position(X=X, pd=pd, vd=vd, ad=ad, t=t, last_wind_update=t_last_wind_update)
            self.t_posctrl += self.params['dt_posctrl']
            self.T_sp = T_sp
            self.q_sp = q_sp
        T_sp = self.T_sp
        q_sp = self.q_sp
        if t >= self.t_attctrl:      # 姿态控制器：获取期望角速度 w_sp
            w_sp = self.attitude(q=X[3:7], q_sp=q_sp)
            self.t_attctrl += self.params['dt_attctrl']
        if t >= self.t_angratectrl:  # 角速度控制器：获取期望力矩 torque_sp
            torque_sp = self.angrate(w=X[10:], w_sp=w_sp, dt=self.params['dt_angratectrl'])
            u = self.mixer(torque_sp=torque_sp, T_sp=T_sp)  # 将期望力矩和期望总推力转换为电机控制输入 u
            self.t_angratectrl += self.params['dt_angratectrl']
        return u