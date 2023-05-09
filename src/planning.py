import numpy as np
import math
from pydrake.all import (
    RotationMatrix,
    MathematicalProgram,
    Solve
)


class PPTrajectory:
    def __init__(self, sample_times, num_vars, degree, continuity_degree):
        self.sample_times = sample_times
        self.n = num_vars
        self.degree = degree

        self.prog = MathematicalProgram()
        self.coeffs = []
        for i in range(len(sample_times)):
            self.coeffs.append(
                self.prog.NewContinuousVariables(num_vars, degree + 1, "C")
            )
        self.result = None

        # Add continuity constraints
        for s in range(len(sample_times) - 1):
            trel = sample_times[s + 1] - sample_times[s]
            coeffs = self.coeffs[s]
            for var in range(self.n):
                for deg in range(continuity_degree + 1):
                    # Don't use eval here, because I want left and right
                    # values of the same time
                    left_val = 0
                    for d in range(deg, self.degree + 1):
                        left_val += (
                                coeffs[var, d]
                                * np.power(trel, d - deg)
                                * math.factorial(d)
                                / math.factorial(d - deg)
                        )
                    right_val = self.coeffs[s + 1][var, deg] * math.factorial(
                        deg
                    )
                    self.prog.AddLinearConstraint(left_val == right_val)

        # Add cost to minimize highest order terms
        for s in range(len(sample_times) - 1):
            self.prog.AddQuadraticCost(
                np.eye(num_vars),
                np.zeros((num_vars, 1)),
                self.coeffs[s][:, -1],
            )

    def eval(self, t, derivative_order=0):
        if derivative_order > self.degree:
            return 0

        s = 0
        while s < len(self.sample_times) - 1 and t >= self.sample_times[s + 1]:
            s += 1
        trel = t - self.sample_times[s]

        if self.result is None:
            coeffs = self.coeffs[s]
        else:
            coeffs = self.result.GetSolution(self.coeffs[s])

        deg = derivative_order
        val = 0 * coeffs[:, 0]
        for var in range(self.n):
            for d in range(deg, self.degree + 1):
                val[var] += (
                        coeffs[var, d]
                        * np.power(trel, d - deg)
                        * math.factorial(d)
                        / math.factorial(d - deg)
                )

        return val

    def add_constraint(self, t, derivative_order, lb, ub=None):
        """Adds a constraint of the form d^deg lb <= x(t) / dt^deg <= ub."""
        if ub is None:
            ub = lb

        assert derivative_order <= self.degree
        val = self.eval(t, derivative_order)
        self.prog.AddLinearConstraint(val, lb, ub)

    def generate(self):
        self.result = Solve(self.prog)
        assert self.result.is_success()


# Thanks Bernhard for presenting a sane way to do the computation!
# (We are using notation from Mellinger's thesis for variable names)
# Notes from Bernhard:

GRAVITY = np.array([0.0, 0.0, -9.81])


class DifferentialFlatness:
    def __int__(self, load_mass, spring_constant, force_constant, moment_constant, arm_length, quad_mass, quad_inertia):
        self.load_mass = load_mass
        self.kF = force_constant
        self.spring_constant = spring_constant
        self.kM = moment_constant
        self.arm_length = arm_length
        self.quad_mass = quad_mass
        self.quad_inertia = quad_inertia

    def compute_quad_state_and_control_from_output(self,
                                                   sigma,
                                                   sigma_dt,
                                                   sigma_ddt,
                                                   sigma_dddt,
                                                   sigma_ddddt):
        # COMPUTE ROLL PITCH YAW
        # compute z axis of body frame
        t = np.array(sigma[0:3] - GRAVITY)
        z_b = t / np.linalg.norm(t)

        u_thrust = self.quad_mass * np.linalg.norm(t)

        # x-component rotated inertial frame (world frame) by angle yaw
        x_c = np.array([np.cos(sigma[3]), np.sin(sigma[3]), 0.0])

        # find the rest of body frame basis vectors
        y_b_unnormed = np.cross(z_b, x_c)
        y_b = y_b_unnormed / np.linalg.norm(y_b_unnormed)
        x_b = np.cross(y_b, z_b)

        # construct a rotation matrix and then convert to rpy
        R_WB = RotationMatrix(R=np.hstack([
            x_b.reshape(-1, 1), y_b.reshape(-1, 1), z_b.reshape(-1, 1)
        ]))
        drake_rpy = R_WB.ToRollPitchYaw()
        rpy = np.array([drake_rpy.roll_angle(), drake_rpy.pitch_angle(), drake_rpy.yaw_angle()])

        # COMPUTE ANGULAR VELOCITY omega = [p q r]
        h_omega = self.quad_mass / u_thrust * (sigma_dddt[0:3] - (np.dot(z_b, sigma_dddt[0:3]) * z_b))
        p = -np.dot(h_omega, y_b)
        q = np.dot(h_omega, x_b)
        r = sigma_dt[2] * z_b[2]
        omega = np.array([p, q, r])

        # COMPUTE ANGULAR ACCELERATION
        temp = np.cross(omega, np.cross(omega, z_b))
        h_alpha = self.quad_mass / u_thrust * (sigma_ddddt[0:3] - (np.dot(z_b, sigma_ddddt[0:3])) * z_b) \
                  + (-temp + np.dot(z_b, temp) * z_b) \
                  - (2 / u_thrust) * z_b.dot(self.quad_mass * sigma_dddt[0:3]) * np.cross(omega, z_b)

        p_dt = -np.dot(h_alpha, y_b)
        q_dt = np.dot(h_alpha, x_b)

        omega_dt = np.array([p_dt, q_dt, 0.0])  # TODO: actually solve for r_dt if we actually need it
        u_moments = np.dot(self.quad_inertia, omega_dt) + np.cross(omega, np.dot(self.quad_inertia, omega))

        # convert thrust and moments to propeller speeds
        u_thrust_moments = np.concatenate(
            [np.array([u_thrust]), u_moments])
        prop_speeds_to_thrust_and_moments = np.array([
            [self.kF, self.kF, self.kF, self.kF],
            [0.0, self.kF * self.arm_length, 0.0, -self.kF * self.arm_length],
            [-self.kF * self.arm_length, 0.0, self.kF * self.arm_length, 0.0],
            [self.kM, -self.kM, self.kM, -self.kM]
        ])
        thrust_and_moments_to_prop_speeds = np.linalg.inv(prop_speeds_to_thrust_and_moments)
        u_all = np.dot(thrust_and_moments_to_prop_speeds, u_thrust_moments)

        return sigma[0:3], rpy, sigma_dt[0:3], omega, u_all

    def compute_point_mass_state_and_control_from_output(
            self,
            mass_position_ds,
            tension_forces_two_to_n_ds,
            yaws_ds
    ):
        """
        mass_positiosn_ds is a 7-tuple, indexed by derivative order, then by xyz
        tension_forces_two_to_n_ds is a 5-tuple, indexed by derivative order. Elements are 2D arrays,
        first index is quad # 2->n. We assume that the forces are PULLING on the mass load
        yaws_ds is a 5-tuple, indexed by derivative order. Elements are 1D arrays, first index is quad # 1->n
        """

        assert len(mass_position_ds) == 7
        assert len(tension_forces_two_to_n_ds) == 5
        assert len(yaws_ds) == 5

        # compute the remaining tension left with force balance
        sum_tension_forces_two_to_n_ds = np.sum(
            np.array(
                tension_forces_two_to_n_ds
            ), axis=1
        )
        tension_forces_one_ds = self.load_mass * mass_position_ds[2:] - sum_tension_forces_two_to_n_ds
        tension_forces_one_ds[0] -= self.load_mass * GRAVITY

        tension_forces_ds = [np.vstack([first, two_to_n])
                             for first, two_to_n
                             in zip(np.split(tension_forces_one_ds, 5), tension_forces_two_to_n_ds)]

        # next... compute the unit vectors and tensions
        tension_forces = tension_forces_ds[0]

        # column here to batch mult/div ops and flatten later
        # Solving for zeroth order force data
        tensions = np.linalg.norm(tension_forces, axis=1).reshape(-1, 1)
        dirs = tension_forces / tensions

        # Solving for first order force data
        tension_forces_dt = tension_forces_ds[1]
        tensions_dt = _stacked_dot_prod(tension_forces_dt, dirs)  # `vectorized dot product across all forces'
        dirs_dt = (tension_forces_dt - tensions_dt * dirs) / tensions

        # Solving for second order force data
        tension_forces_ddt = tension_forces_ds[2]
        tensions_ddt = _stacked_dot_prod(tension_forces_ddt, dirs) + _stacked_dot_prod(tension_forces_dt, dirs_dt)
        dirs_ddt = (
                           tension_forces_ddt
                           - tensions_ddt * dirs
                           - 2 * tensions_dt * dirs_dt
                   ) / tensions

        # solving for third order force data
        tension_forces_dddt = tension_forces_ds[3]
        tensions_dddt = _stacked_dot_prod(tension_forces_dddt, dirs) \
                        + 2 * _stacked_dot_prod(tension_forces_ddt, dirs_dt) \
                        + _stacked_dot_prod(tension_forces_dt, dirs_ddt)
        dirs_dddt = (
                            tension_forces_dddt
                            - tensions_dddt * dirs
                            - 3 * tensions_ddt * dirs_dt
                            - 3 * tensions_dt * dirs_ddt
                    ) / tensions

        tension_forces_ddddt = tension_forces_ds[4]
        tensions_ddddt = _stacked_dot_prod(tension_forces_ddddt, dirs) \
                         + 3 * _stacked_dot_prod(tension_forces_dddt, dirs_dt) \
                         + 3 * _stacked_dot_prod(tension_forces_ddt, dirs_ddt) \
                         + _stacked_dot_prod(tension_forces_dt, dirs_dddt)
        dirs_ddddt = (
                             tensions_ddddt
                             - tensions_ddddt * dirs
                             - 4 * tensions_dddt * dirs_dt
                             - 6 * tensions_ddt * dirs_ddt
                             - 4 * tensions_dt * dirs_dddt
                     ) / tensions

        # next, we solve for the poses of the quad coms using Hooke's spring law. here, we make the assumption
        # that the forces are PULLING on the mass load.

        # since the general form of the equation is the same, we solve in a list comprehension
        quad_position_ds = [
            tension_forces_di / self.spring_constant + mass_position_di
            for tension_forces_di, mass_position_di in zip(tension_forces_ds, mass_position_ds)
        ]

        # next, we solve the quad control inputs using the quad differential flatness

        # currently, the arrays are indexed by [order of deriv] X [number quad] X [state index]
        # we'll swap the indices so that we can iterate over them (i.e. [number quad] X [order of deriv] X [state index])
        quad_pos_indexed_by_quad = np.swapaxes(np.array(quad_position_ds), 0, 1)
        yaws_indexed_by_quad = np.swapaxes(np.expand_dims(yaws_ds, 2), 0, 1)

        n_quad = len(quad_pos_indexed_by_quad)
        quad_pos_all = np.zeros((n_quad, 3))
        quad_rpy_all = np.zeros((n_quad, 3))
        quad_vel_all = np.zeros((n_quad, 3))
        quad_omega_all = np.zeros((n_quad, 3))
        quad_us_all = np.zeros((n_quad, 4))

        for i, (quad_i_pos_ds, quad_i_yaw_ds) in enumerate(zip(quad_pos_indexed_by_quad, yaws_indexed_by_quad)):
            sigma_ds = np.hstack([quad_i_pos_ds, quad_i_yaw_ds])
            quad_pos_all[i, :], quad_rpy_all[i, :], quad_vel_all[i, :], quad_omega_all[i, :], quad_us_all[i, :] \
                = self.compute_quad_state_and_control_from_output(*sigma_ds)

        return quad_pos_all, quad_rpy_all, quad_vel_all, quad_omega_all, quad_us_all

    def state_to_output(self, mass_pos, quad_poses, quad_yaws):
        """
        Given a state, (assuming ZERO YAW on quads), output corresponding output.
        The Yaw is zero since the details in Mellinger's thesis were missing to implement
        them faithfully.

        We're assuming that quad_poses are indexed num quad (0,..., n-1) X quad pos (XYZ)
        and load mass is indexed as (3,) vector (XYZ)
        """
        tension_output = self.spring_constant * (quad_poses - mass_pos)
        return mass_pos, tension_output, quad_yaws


def straight_trajectory_from_outputA_to_outputB(mass_output_A,
                                                tension_output_A,
                                                yaw_output_A,
                                                mass_output_B,
                                                yaw_output_B,
                                                tension_output_B,
                                                tf=3.0, delta_t=0.1):
    """
    Mass output: (3,) (XYZ)
    Tension output index: num_quad (0, ..., n-2) X force component (XYZ)
    Yaw output index: num_quad (0, ..., n-1)
    We not include yaw output since
    """

    assert mass_output_A.shape == (3,) and mass_output_B.shape == (3,)
    assert tension_output_A.shape[1] == 3 and tension_output_B.shape[1] == 3
    assert np.isclose(yaw_output_A, yaw_output_B)  # we don't handle changes in yaw that well

    n_output = 3 + tension_output_A.size
    n_quad = tension_output_A.shape[0] + 1

    zpp_traj = PPTrajectory(
        sample_times=np.linspace(0, tf, 3),
        num_vars=n_output,
        degree=7,
        continuity_degree=6
    )

    zpp_traj.add_constraint(t=0, derivative_order=0,
                            lb=np.concatenate(
                                [mass_output_A, tension_output_A.flatten(order='C')]))  # row-major flatten
    zpp_traj.add_constraint(t=0, derivative_order=1, lb=np.zeros(n_output))
    zpp_traj.add_constraint(t=0, derivative_order=2, lb=np.zeros(n_output))

    zpp_traj.add_constraint(t=tf, derivative_order=0,
                            lb=np.concatenate([mass_output_B, tension_output_B.flatten(order='C')]))
    zpp_traj.add_constraint(t=tf, derivative_order=1, lb=np.zeros(n_output))
    zpp_traj.add_constraint(t=tf, derivative_order=2, lb=np.zeros(n_output))

    zpp_traj.generate()

    n_t = math.ceil(tf / delta_t)
    ts = np.linspace(0, tf, n_t)

    # construct output trajectories in the format that compute_...state_and_control() expects.
    # indexing for mass: time X derivative_order X mass state var (XYZ)
    # indexing for tensions: time X derivative_order X quad # (1, ..., n) X tension vect state var (XYZ)
    # indexing for yaws: time X derivative_order X quad # (0, ..., n)

    mass_position_ds = np.zeros((n_t, 7, 3))
    tension_forces_ds = np.zeros((n_t, 5, n_quad - 1, 3))
    yaws_ds = np.zeros((n_t, 5, n_quad))
    yaws_ds[0, :] = yaw_output_A

    for t in range(ts):
        # confirm that this reshape is correct

        for i in range(7):
            mass_position_ds[t, i, :] = zpp_traj.eval(t, derivative_order=i)[:3]

        for i in range(5):
            tension_forces_ds[t, i, :, :] = zpp_traj.eval(t, derivative_order=i)[3:]

    return mass_position_ds, tension_forces_ds, yaws_ds


# an interesting test case: two quads stay still, but third quad will pull the mass away
def demo_traj_for_three_quads(load_mass, kF, kM, arm_length, quad_mass, quad_inertia, spring_constant, num_steps=100):
    """Moves 1 meter in the y direction, with increments split over num_steps (i'm not sure
    at what timescale we are using so I included that degree of freedom so you can
    set a reasonable timestep)
    """
    tension_output_trajs = [
        np.array([[1.0, 0.0, 1.0],
                  [-1.0, 0.0, 1.0]]),
        np.zeros((2, 3)),
        np.zeros((2, 3)),
        np.zeros((2, 3)),
        np.zeros((2, 3))
    ]

    yaws_output_trajs = [
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3)
    ]

    mass_output_trajs = np.array([
        [
            np.array([0.0, i * 1.0 / num_steps, 0.0]),
            np.array([0.0, 1.0 / num_steps, 0.0]),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3)
        ] for i in range(num_steps)
    ])

    quad_all_pos_traj, quad_all_rpy_traj, quad_all_vel_traj, quad_all_omega_traj, quad_all_us_traj = [], [], [], [], []
    for mass_output, tension_output, yaws_output in zip(mass_output_trajs, tension_output_trajs, yaws_output_trajs):
        quad_all_pos, quad_all_rpy, quad_all_vel, quad_all_omega, quad_all_us = compute_point_mass_state_and_control_from_output(
            load_mass, kF, kM, arm_length, quad_mass, quad_inertia,
            spring_constant, mass_output, tension_output, yaws_output)

        quad_all_pos_traj.append(quad_all_pos)
        quad_all_rpy_traj.append(quad_all_rpy)
        quad_all_vel_traj.append(quad_all_vel)
        quad_all_omega_traj.append(quad_all_omega)
        quad_all_us_traj.append(quad_all_us)

    return quad_all_pos_traj, quad_all_rpy_traj, quad_all_vel_traj, quad_all_omega_traj, quad_all_us_traj


def _stacked_dot_prod(stack_of_v1, stack_of_v2):
    return np.sum(stack_of_v1 * stack_of_v2, axis=1).reshape(-1, 1)


# for debugging purposes
if __name__ == '__main__':
    # system constants
    load_dummy_mass = 1
    quad_dummy_mass = 0.775
    quad_dummy_inertia = np.diag([0.0015, 0.0025, 0.0035])
    dummy_kF = 1.0
    dummy_kM = 0.0245
    dummy_arm_length = 0.15
    spring_dummy_constant = 10

    # we'll test a single timeframe
    # assume we have three quads, but two of them are balancing the mass already (expecting zero force on the remaining)
    mass_output = (
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3)
    )

    tension_forces_output = (
        np.array([[1.0, 0.0, 1.0],
                  [-1.0, 0.0, 1.0]]),
        np.zeros((2, 3)),
        np.zeros((2, 3)),
        np.zeros((2, 3)),
        np.zeros((2, 3))
    )

    yaws_output = (
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3)
    )

    quad_pos_all, quad_rpy_all, quad_vel_all, quad_omega_all, quad_us_all \
        = compute_point_mass_state_and_control_from_output(load_dummy_mass, dummy_kF, dummy_kM, dummy_arm_length,
                                                           quad_dummy_mass, quad_dummy_inertia,
                                                           spring_dummy_constant,
                                                           mass_output, tension_forces_output, yaws_output)

    # then we can try something a little less trivial (i.e. move the quads in a circle around obj still stable
    print('stable equilibrium')
    print('quad_pos_all: \n' + str(quad_pos_all))
    print('quad_vel_all: \n' + str(quad_vel_all))
    print('quad_rpy_all: \n' + str(quad_rpy_all))
    print('quad_omega_all: \n' + str(quad_omega_all))
    print('quad_us_all: \n' + str(quad_us_all))

    mass_output = (
        np.zeros(3),
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3)
    )

    two_pi_over_three = np.pi * 2 / 3
    tension_forces_output = (
        (9.81 + 1) / 3 * np.array([[np.cos(0), np.sin(0), 1.0],
                                   [np.cos(two_pi_over_three), np.sin(two_pi_over_three), 1.0]]),
        np.zeros((2, 3)),
        np.array([[0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0]]),
        np.zeros((2, 3)),
        np.zeros((2, 3))
    )

    yaws_output = (
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3)
    )

    quad_pos_all, quad_rpy_all, quad_vel_all, quad_omega_all, quad_us_all \
        = compute_point_mass_state_and_control_from_output(load_dummy_mass, dummy_kF, dummy_kM, dummy_arm_length,
                                                           quad_dummy_mass, quad_dummy_inertia,
                                                           spring_dummy_constant,
                                                           mass_output, tension_forces_output, yaws_output)
    print('\naccelerating upward uniformally')
    print('quad_pos_all: \n' + str(quad_pos_all))
    print('quad_vel_all: \n' + str(quad_vel_all))
    print('quad_rpy_all: \n' + str(quad_rpy_all))
    print('quad_omega_all: \n' + str(quad_omega_all))
    print('quad_us_all: \n' + str(quad_us_all))

    mass_output = (
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3)
    )

    two_pi_over_three = np.pi * 2 / 3
    tension_forces_output = (
        9.81 / 3 * np.array([[np.cos(0), np.sin(0), 1.0],
                             [np.cos(two_pi_over_three), np.sin(two_pi_over_three), 1.0]]),
        10 * np.array([[-np.sin(0), np.cos(0), 0.0],
                       [-np.sin(two_pi_over_three), np.cos(two_pi_over_three), 0.0]]),
        np.zeros((2, 3)),
        np.zeros((2, 3)),
        np.zeros((2, 3))
    )

    yaws_output = (
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3)
    )

    quad_pos_all, quad_rpy_all, quad_vel_all, quad_omega_all, quad_us_all \
        = compute_point_mass_state_and_control_from_output(load_dummy_mass, dummy_kF, dummy_kM, dummy_arm_length,
                                                           quad_dummy_mass, quad_dummy_inertia,
                                                           spring_dummy_constant,
                                                           mass_output, tension_forces_output, yaws_output)
    print('\ncircling')
    print('quad_pos_all: \n' + str(quad_pos_all))
    print('quad_vel_all: \n' + str(quad_vel_all))
    print('quad_rpy_all: \n' + str(quad_rpy_all))
    print('quad_omega_all: \n' + str(quad_omega_all))
    print('quad_us_all: \n' + str(quad_us_all))

    demo_traj_for_three_quads(load_dummy_mass, dummy_kF, dummy_kM, dummy_arm_length, quad_dummy_mass,
                              quad_dummy_inertia,
                              spring_dummy_constant)
