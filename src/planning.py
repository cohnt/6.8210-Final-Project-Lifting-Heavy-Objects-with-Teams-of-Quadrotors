import numpy as np
import math
from pydrake.all import (
    RotationMatrix,
    MathematicalProgram,
    Solve,
    MosekSolver
)

GRAVITY = np.array([0.0, 0.0, -9.81])


# DEBUG LIST:
# [x] Check (for complicated trajectories) that pushing sol back to output yields correct input
# [ ] If not: start by checking tension math
# [ ] Then continue into quad math

class PPTrajectory:
    def __init__(self, sample_times, num_vars, degree, continuity_degree):
        self.sample_times = sample_times
        self.n = num_vars
        self.degree = degree

        self.prog = MathematicalProgram()
        self.coeffs = []
        for i in range(len(sample_times)):
            # decision vars are pnoms + derivs, deriv relation added as constraint
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
        solver = MosekSolver()
        self.result = solver.Solve(self.prog)
        assert self.result.is_success(), str(self.result.get_solver_details().solution_status)


# Thanks Bernhard for presenting a sane way to do the computation!
# (We are using notation from Mellinger's thesis for variable names)
# Notes from Bernhard:


class DifferentialFlatness:
    def __init__(self, load_mass, cable_length, spring_constant, force_constant, moment_constant, arm_length, quad_mass,
                 quad_inertia):
        self.load_mass = load_mass
        self.cable_length = cable_length
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
        t = np.array(sigma_ddt[0:3] - GRAVITY)
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
        ]).T)
        drake_rpy = R_WB.ToRollPitchYaw()
        rpy = np.array([drake_rpy.roll_angle(), drake_rpy.pitch_angle(), drake_rpy.yaw_angle()])

        # COMPUTE ANGULAR VELOCITY omega = [p q r]
        h_omega = self.quad_mass / u_thrust * (sigma_dddt[0:3] - (np.dot(z_b, sigma_dddt[0:3]) * z_b))
        p = -np.dot(h_omega, y_b)
        q = np.dot(h_omega, x_b)
        r = np.dot(sigma_dt[3] * np.array([0.0, 0.0, 1.0]), z_b)
        omega = np.array([p, q, r])

        # COMPUTE ANGULAR ACCELERATION
        temp = np.cross(omega, np.cross(omega, z_b))
        h_alpha = (self.quad_mass / u_thrust) * (sigma_ddddt[0:3] - (np.dot(z_b, sigma_ddddt[0:3])) * z_b) \
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

        # F = K(x_quad - (x_m + dir * length))
        # F^(i) = K(x_quad^(i) - (x_m^(i) + dir^(i) * length))
        dirs_ds = (dirs, dirs_dt, dirs_ddt, dirs_dddt, dirs_ddddt)

        quad_position_ds = [
            tension_forces_di / self.spring_constant + mass_position_di + dirs_di * self.cable_length
            for tension_forces_di, mass_position_di, dirs_di in zip(tension_forces_ds, mass_position_ds, dirs_ds)
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
        mass_to_quad = quad_poses - mass_pos
        mass_to_quad_dist = np.linalg.norm(mass_to_quad, axis=1).reshape(-1, 1)
        q_mass_to_quad = mass_to_quad / mass_to_quad_dist
        tension_output = (self.spring_constant * (quad_poses - mass_pos - self.cable_length * q_mass_to_quad))
        in_tension_mask = np.array((mass_to_quad_dist >= self.cable_length), dtype=np.float32)
        tension_output *= in_tension_mask
        if not (in_tension_mask > 0).all():
            print('WARNING: some quads not in tension. quads in tension: %s' % str(in_tension_mask))

        return mass_pos, tension_output[1:], quad_yaws

    def state_traj_to_output_traj(self, mass_pos_traj, quad_pos_traj, quad_yaw_traj):
        nt, nquad, nxyz = quad_pos_traj.shape
        assert nxyz == 3

        tension_traj = np.zeros((nt, nquad - 1, nxyz))
        for i, (mass_pos, quad_pos, quad_yaw) in enumerate(zip(mass_pos_traj, quad_pos_traj, quad_yaw_traj)):
            _, tension_traj[i, :, :], _ = self.state_to_output(mass_pos, quad_pos, quad_yaw)

        return mass_pos_traj, tension_traj, quad_yaw_traj

    def output_traj_to_state_traj(self, mass_output_trajs, tension_output_trajs, yaws_output_trajs):
        quad_all_pos_traj, quad_all_rpy_traj, quad_all_vel_traj, quad_all_omega_traj, quad_all_us_traj = [], [], [], [], []
        for mass_output, tension_output, yaws_output in zip(mass_output_trajs, tension_output_trajs, yaws_output_trajs):
            quad_all_pos, quad_all_rpy, quad_all_vel, quad_all_omega, quad_all_us = \
                self.compute_point_mass_state_and_control_from_output(
                    mass_output,
                    tension_output,
                    yaws_output)

            quad_all_pos_traj.append(quad_all_pos)
            quad_all_rpy_traj.append(quad_all_rpy)
            quad_all_vel_traj.append(quad_all_vel)
            quad_all_omega_traj.append(quad_all_omega)
            quad_all_us_traj.append(quad_all_us)

        return np.array(quad_all_pos_traj), \
            np.array(quad_all_rpy_traj), \
            np.array(quad_all_vel_traj), \
            np.array(quad_all_omega_traj), \
            np.array(quad_all_us_traj), \
            np.array(mass_output_trajs[:, 0, :]), \
            np.array(mass_output_trajs[:, 1, :])

    def demo_traj_for_three_quads(self, num_steps=100):
        """Moves 1 meter in the y direction, with increments split over num_steps (i'm not sure
        at what timescale we are using so I included that degree of freedom so you can
        set a reasonable timestep)
        :param cable_length:
        """

        two_pi_over_three = 2 * np.pi / 3
        tension_output_trajs = np.array([[
            9.81 / 3 * np.array([[np.cos(0), np.sin(0), 1.0],
                                 [np.cos(two_pi_over_three), np.sin(two_pi_over_three), 1.0]]),
            np.zeros((2, 3)),
            np.zeros((2, 3)),
            np.zeros((2, 3)),
            np.zeros((2, 3))
        ] for _ in range(num_steps)])

        yaws_output_trajs = np.array([[
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3)
        ] for _ in range(num_steps)])

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

        quad_all_pos_traj, quad_all_rpy_traj, quad_all_vel_traj, quad_all_omega_traj, quad_all_us_traj, mass_pos_traj, mass_vel_traj = \
            self.output_traj_to_state_traj(mass_output_trajs, tension_output_trajs, yaws_output_trajs)

        mass_quat_traj = [np.array([1, 0, 0, 0]) for _ in range(len(mass_output_trajs))]
        mass_omega_traj = [np.zeros(3) for _ in range(len(mass_output_trajs))]

        return quad_all_pos_traj, quad_all_rpy_traj, quad_all_vel_traj, quad_all_omega_traj, quad_all_us_traj, \
            mass_pos_traj, mass_vel_traj, mass_quat_traj, mass_omega_traj

    def straight_trajectory_from_outputA_to_outputB(self,
                                                    mass_outputA,
                                                    tension_outputA,
                                                    yaw_outputA,
                                                    mass_outputB,
                                                    tension_outputB,
                                                    yaw_outputB,
                                                    tf=3.0, dt=0.1):
        """
        Mass output: (3,) (XYZ)
        Tension output index: num_quad (0, ..., n-2) X force component (XYZ)
        Yaw output index: num_quad (0, ..., n-1)
        We not include yaw output since
        """

        assert mass_outputA.shape == (3,) and mass_outputB.shape == (3,)
        assert tension_outputA.shape[1] == 3 and tension_outputB.shape[1] == 3
        assert np.isclose(yaw_outputA, yaw_outputB).all()  # we don't handle changes in yaw that well

        n_quad = tension_outputA.shape[0] + 1

        mass_traj = PPTrajectory(
            sample_times=np.linspace(0, tf, 2),
            num_vars=3,
            degree=7,
            continuity_degree=6
        )
        mass_traj.add_constraint(t=0, derivative_order=0, lb=mass_outputA)
        mass_traj.add_constraint(t=tf, derivative_order=0, lb=mass_outputB)

        mass_traj.generate()

        n_tension_vars = tension_outputA.size
        tension_traj = PPTrajectory(
            sample_times=np.linspace(0, tf, 2),
            num_vars=n_tension_vars,
            degree=5,
            continuity_degree=4
        )
        tension_traj.add_constraint(t=0, derivative_order=0, lb=tension_outputA.flatten())  # row-major flatten
        tension_traj.add_constraint(t=tf, derivative_order=0, lb=tension_outputB.flatten())

        tension_traj.generate()

        n_t = math.ceil(tf / dt)
        ts = np.linspace(0, tf, n_t)

        # construct output trajectories in the format that compute_...state_and_control() expects.
        # indexing for mass: time X derivative_order X mass state var (XYZ)
        # indexing for tensions: time X derivative_order X quad # (1, ..., n) X tension vect state var (XYZ)
        # indexing for yaws: time X derivative_order X quad # (0, ..., n)

        mass_position_ds = np.zeros((n_t, 7, 3))
        tension_forces_ds = np.zeros((n_t, 5, n_quad - 1, 3))
        yaws_ds = np.zeros((n_t, 5, n_quad))
        yaws_ds[:, 0, :] = np.tile(yaw_outputA, (n_t, 1))

        for t_ix, t in enumerate(ts):
            # confirm that this reshape is correct

            # since we only got linear to work in this case, then only update with the data represented in the polynomial
            for i in range(2):
                mass_position_ds[t_ix, i, :] = mass_traj.eval(t, derivative_order=i)

            for i in range(2):
                tension_forces_ds[t_ix, i, :, :] = tension_traj.eval(t, derivative_order=i).reshape(
                    tension_outputA.shape)

        return mass_position_ds, tension_forces_ds, yaws_ds

    def waypoints_to_output_trajectory_for_three_quads(self, times, mass_waypoints, dt=0.01):
        """
        Two quads will stay above the mass, the third will do the work of moving the trajectory around.
        """
        assert len(times) == len(mass_waypoints)
        n_samples = len(times)
        tf = times[-1]

        two_pi_over_three = 2 * np.pi / 3
        quads_pos_relative_to_mass = self.load_mass * 9.81 / 3 / self.spring_constant * np.array(
            [[np.cos(0), np.sin(0), 1.0],
             [np.cos(two_pi_over_three), np.sin(two_pi_over_three), 1.0],
             [np.cos(2 * two_pi_over_three), np.sin(2 * two_pi_over_three), 1.0]]  # lost through output map
        )

        quads_waypoints = np.repeat(np.expand_dims(mass_waypoints, 1), 3, axis=1) + quads_pos_relative_to_mass
        yaws = np.zeros(3)

        # compute corresponding output map
        output_waypoints = [
            self.state_to_output(mass_waypoints[i], quads_waypoints[i], yaws) for i in range(n_samples)
        ]
        n_tension_vars = output_waypoints[0][1].size
        tension_shape = output_waypoints[0][1].shape

        # run optimization to find the piecewise polynomials
        mass_traj = PPTrajectory(
            sample_times=np.linspace(0, tf, n_samples),
            num_vars=3,
            degree=7,
            continuity_degree=6
        )

        tension_traj = PPTrajectory(
            sample_times=np.linspace(0, tf, n_samples),
            num_vars=n_tension_vars,
            degree=5,
            continuity_degree=4
        )

        mass_traj.add_constraint(t=0, derivative_order=1, lb=np.zeros(3))
        mass_traj.add_constraint(t=0, derivative_order=2, lb=np.zeros(3))
        mass_traj.add_constraint(t=tf, derivative_order=1, lb=np.zeros(3))
        mass_traj.add_constraint(t=tf, derivative_order=2, lb=np.zeros(3))

        tension_traj.add_constraint(t=0, derivative_order=1, lb=np.zeros(n_tension_vars))
        tension_traj.add_constraint(t=0, derivative_order=2, lb=np.zeros(n_tension_vars))
        tension_traj.add_constraint(t=tf, derivative_order=1, lb=np.zeros(n_tension_vars))
        tension_traj.add_constraint(t=tf, derivative_order=2, lb=np.zeros(n_tension_vars))

        for t, (mass_wp, tension_wp, _) in zip(times, output_waypoints):
            mass_traj.add_constraint(t=t, derivative_order=0, lb=mass_wp)
            tension_traj.add_constraint(t=t, derivative_order=0, lb=tension_wp.flatten(order='C'))

        try:
            mass_traj.generate()
        except AssertionError:
            print('Could not find a piecewise-polynomial to fit the mass_traj.')

        try:
            tension_traj.generate()
        except AssertionError:
            print('Could not find a piecewise-polynomial to fit the tension_traj.')

        # evaluate the polynomials to get the output trajectory
        n_t = math.ceil(tf / dt)
        ts = np.linspace(0, tf, n_t)

        mass_position_ds = np.zeros((n_t, 7, 3))
        tension_forces_ds = np.zeros((n_t, 5, 2, 3))
        yaws_ds = np.zeros((n_t, 5, 3))
        yaws_ds[:, 0, :] = np.tile(yaws, (n_t, 1))

        for t_ix, t in enumerate(ts):
            # confirm that this reshape is correct

            for i in range(7):
                mass_position_ds[t_ix, i, :] = mass_traj.eval(t, derivative_order=i)

            for i in range(5):
                tension_forces_ds[t_ix, i, :, :] = tension_traj.eval(t, derivative_order=i).reshape(
                    output_waypoints[0][1].shape, order='C')

        return mass_position_ds, tension_forces_ds, yaws_ds


def _stacked_dot_prod(stack_of_v1, stack_of_v2):
    return np.sum(stack_of_v1 * stack_of_v2, axis=1).reshape(-1, 1)


# for debugging purposes
if __name__ == '__main__':
    # system constants
    dummy_cable_length = 1
    load_dummy_mass = 1
    quad_dummy_mass = 0.775
    quad_dummy_inertia = np.diag([0.0015, 0.0025, 0.0035])
    dummy_kF = 1.0
    dummy_kM = 0.0245
    dummy_arm_length = 0.15
    spring_dummy_constant = 1

    output_backer = DifferentialFlatness(load_dummy_mass, dummy_cable_length, spring_dummy_constant, dummy_arm_length,
                                         dummy_kF,
                                         dummy_kM, quad_dummy_mass, quad_dummy_inertia)

    # we'll test a single timeframe
    # assume we have three quads, but two of them are balancing the mass already (expecting zero force on the remaining)
    # mass_output = (
    #     np.zeros(3),
    #     np.zeros(3),
    #     np.zeros(3),
    #     np.zeros(3),
    #     np.zeros(3),
    #     np.zeros(3),
    #     np.zeros(3)
    # )
    #
    # tension_forces_output = (
    #     np.array([[1.0, 0.0, 1.0],
    #               [-1.0, 0.0, 1.0]]),
    #     np.zeros((2, 3)),
    #     np.zeros((2, 3)),
    #     np.zeros((2, 3)),
    #     np.zeros((2, 3))
    # )
    #
    # yaws_output = (
    #     np.zeros(3),
    #     np.zeros(3),
    #     np.zeros(3),
    #     np.zeros(3),
    #     np.zeros(3)
    # )
    #
    # quad_pos_all, quad_rpy_all, quad_vel_all, quad_omega_all, quad_us_all \
    #     = output_backer.compute_point_mass_state_and_control_from_output(mass_output,
    #                                                                      tension_forces_output,
    #                                                                      yaws_output)
    #
    # # then we can try something a little less trivial (i.e. move the quads in a circle around obj still stable
    # print('stable equilibrium')
    # print('quad_pos_all: \n' + str(quad_pos_all))
    # print('quad_vel_all: \n' + str(quad_vel_all))
    # print('quad_rpy_all: \n' + str(quad_rpy_all))
    # print('quad_omega_all: \n' + str(quad_omega_all))
    # print('quad_us_all: \n' + str(quad_us_all))
    #
    # mass_output = (
    #     np.zeros(3),
    #     np.zeros(3),
    #     np.array([0.0, 0.0, 1.0]),
    #     np.zeros(3),
    #     np.zeros(3),
    #     np.zeros(3),
    #     np.zeros(3)
    # )
    #
    # two_pi_over_three = np.pi * 2 / 3
    # tension_forces_output = (
    #     (9.81 + 1) / 3 * np.array([[np.cos(0), np.sin(0), 1.0],
    #                                [np.cos(two_pi_over_three), np.sin(two_pi_over_three), 1.0]]),
    #     np.zeros((2, 3)),
    #     np.array([[0.0, 0.0, 0.0],
    #               [0.0, 0.0, 0.0]]),
    #     np.zeros((2, 3)),
    #     np.zeros((2, 3))
    # )
    #
    # yaws_output = (
    #     np.zeros(3),
    #     np.zeros(3),
    #     np.zeros(3),
    #     np.zeros(3),
    #     np.zeros(3)
    # )
    #
    # quad_pos_all, quad_rpy_all, quad_vel_all, quad_omega_all, quad_us_all \
    #     = output_backer.compute_point_mass_state_and_control_from_output(mass_output,
    #                                                                      tension_forces_output,
    #                                                                      yaws_output)
    # print('\naccelerating upward uniformally')
    # print('quad_pos_all: \n' + str(quad_pos_all))
    # print('quad_vel_all: \n' + str(quad_vel_all))
    # print('quad_rpy_all: \n' + str(quad_rpy_all))
    # print('quad_omega_all: \n' + str(quad_omega_all))
    # print('quad_us_all: \n' + str(quad_us_all))
    #
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
    #
    # quad_pos_all, quad_rpy_all, quad_vel_all, quad_omega_all, quad_us_all \
    #     = output_backer.compute_point_mass_state_and_control_from_output(mass_output,
    #                                                                      tension_forces_output,
    #                                                                      yaws_output)
    # print('\ncircling')
    # print('quad_pos_all: \n' + str(quad_pos_all))
    # print('quad_vel_all: \n' + str(quad_vel_all))
    # print('quad_rpy_all: \n' + str(quad_rpy_all))
    # print('quad_omega_all: \n' + str(quad_omega_all))
    # print('quad_us_all: \n' + str(quad_us_all))
    #
    # # testing for n = 3 quads using the last circling operation
    mass_posA = mass_output[0]
    tension_forces = tension_forces_output[0]
    yaws = yaws_output[0]

    # should be unit tests, but just checking output solve/output map consistency here
    mass_posB = mass_output[1] + np.array([1.0, 0.0, 0.0])
    mass_pos_ds, tension_forces_ds, yaws_ds = output_backer.straight_trajectory_from_outputA_to_outputB(
        mass_posA, tension_forces, yaws, mass_posB, tension_forces, yaws, tf=5)
    quad_all_pos_traj, \
        quad_all_rpy_traj, \
        quad_all_vel_traj, \
        quad_all_omega_traj, \
        quad_all_us_traj, \
        mass_state_pos_traj, \
        mass_state_vel_traj = output_backer.output_traj_to_state_traj(mass_pos_ds,
                                                                      tension_forces_ds,
                                                                      yaws_ds)
    mass_reoutput, tension_reoutput, yaws_reoutput = output_backer.state_traj_to_output_traj(
        mass_state_pos_traj,
        quad_all_pos_traj,
        quad_all_rpy_traj[:, :, 2]
    )
    assert np.all(np.isclose(mass_reoutput, mass_pos_ds[:, 0, :]))
    assert np.all(np.isclose(tension_reoutput, tension_forces_ds[:, 0, :, :]))
    assert np.all(np.isclose(yaws_reoutput, yaws_ds[:, 0, :]))

    times = [0.0, 3.0 / 3, 2 * 3.0 / 3, 3.0]
    waypoints = np.array([
    [0.0, 0.0, 0.0],
    [0.2, 0.5, 0.0],
    [-0.2, 0.7, 0.0],
    [0.0, 1.0, 0.0]
    ])
    mass_pos_ds, tension_forces_ds, yaws_ds = output_backer.waypoints_to_output_trajectory_for_three_quads(times,
                                                                                                           waypoints)
    quad_all_pos_traj, \
        quad_all_rpy_traj, \
        quad_all_vel_traj, \
        quad_all_omega_traj, \
        quad_all_us_traj, \
        mass_state_pos_traj, \
        mass_state_vel_traj = output_backer.output_traj_to_state_traj(mass_pos_ds,
                                                                      tension_forces_ds,
                                                                      yaws_ds)
    mass_reoutput, tension_reoutput, yaws_reoutput = output_backer.state_traj_to_output_traj(
        mass_state_pos_traj,
        quad_all_pos_traj,
        quad_all_rpy_traj[:, :, 2]
    )
    assert np.all(np.isclose(mass_reoutput, mass_pos_ds[:, 0, :]))
    assert np.all(np.isclose(tension_reoutput, tension_forces_ds[:, 0, :, :]))
    assert np.all(np.isclose(yaws_reoutput, yaws_ds[:, 0, :]))
