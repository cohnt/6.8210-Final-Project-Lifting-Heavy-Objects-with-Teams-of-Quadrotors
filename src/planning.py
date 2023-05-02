import numpy as np
import pydot
import pydrake
from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    Parser,
    Propeller,
    PropellerInfo,
    RigidTransform,
    StartMeshcat,
    MeshcatVisualizer,
    SceneGraph,
    Simulator,
    AddMultibodyPlantSceneGraph,
    LeafSystem,
    LeafSystem_,
    ExternallyAppliedSpatialForce,
    ExternallyAppliedSpatialForce_,
    TemplateSystem,
    AbstractValue,
    SpatialForce,
    SpatialForce_,
    SpatialInertia,
    UnitInertia,
    CollisionFilterDeclaration,
    GeometrySet,
    RotationMatrix,
    Sphere
)
from pydrake.examples import (
    QuadrotorGeometry
)
from IPython.display import display, SVG, Image

from underactuated.scenarios import AddFloatingRpyJoint

from tensile import TensileForce, SpatialForceConcatinator

from pydrake.systems.primitives import Adder

# Thanks Bernhard for presenting a sane way to do the computation!
# (We are using notation from Mellinger's thesis for variable names)
# Notes from Bernhard:

GRAVITY = np.array([0.0, 0.0, -9.81])


def compute_quad_state_and_control_from_output(mass, inertia, sigma, sigma_dt, sigma_ddt, sigma_dddt, sigma_ddddt):
    # COMPUTE ROLL PITCH YAW
    # compute z axis of body frame
    t = np.array(sigma[0:3] - GRAVITY)
    z_b = t / np.linalg.norm(t)

    u_first = mass * np.linalg.norm(t)

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
    h_omega = mass / u_first * (sigma_dddt[0:3] - (np.dot(z_b, sigma_dddt[0:3]) * z_b))
    p = -np.dot(h_omega, y_b)
    q = np.dot(h_omega, x_b)
    r = sigma_dt[2] * z_b[2]
    omega = np.array([p, q, r])

    # COMPUTE ANGULAR ACCELERATION
    temp = np.cross(omega, np.cross(omega, z_b))
    h_alpha = mass / u_first * (sigma_ddddt[0:3] - (np.dot(z_b, sigma_ddddt[0:3])) * z_b) \
              + (-temp + np.dot(z_b, temp) * z_b) \
              - (2 / u_first) * z_b.dot(mass * sigma_dddt[0:3]) * np.cross(omega, z_b)

    p_dt = -np.dot(h_alpha, y_b)
    q_dt = np.dot(h_alpha, x_b)

    omega_dt = np.array([p_dt, q_dt, 0.0])  # TODO: actually solve for r_dt if we actually need it
    u_rest = np.dot(inertia, omega_dt) + np.cross(omega, np.dot(inertia, omega))

    u_all = np.concatenate([np.array([u_first]), u_rest]) # TODO: this computation is a little suss... try again with
                                                          # actual quad inertias... unit inertia doesn't give sane results
    return sigma[0:3], rpy, sigma_dt[0:3], omega, u_all


def compute_point_mass_state_and_control_from_output(
        load_mass,
        quad_mass,
        quad_inertia,
        spring_constant,
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
    tension_forces_one_ds = load_mass * mass_position_ds[2:] + sum_tension_forces_two_to_n_ds
    tension_forces_one_ds[0] -= load_mass * GRAVITY # TODO: on sanity trajs, check if there is a sign flip here

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
        tension_forces_di / spring_constant + mass_position_di
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
            = compute_quad_state_and_control_from_output(quad_mass, quad_inertia, *sigma_ds)

    return quad_pos_all, quad_rpy_all, quad_vel_all, quad_omega_all, quad_us_all


def _stacked_dot_prod(stack_of_v1, stack_of_v2):
    return np.sum(stack_of_v1 * stack_of_v2, axis=1).reshape(-1, 1)


def output_map_factory():
    pass  # TODO, returns a function


# for debugging purposes
if __name__ == '__main__':
    # for ease:
    load_dummy_mass = 1
    quad_dummy_mass = 1
    quad_dummy_inertia = np.eye(3)
    spring_dummy_constant = 1

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
        = compute_point_mass_state_and_control_from_output(load_dummy_mass, quad_dummy_mass, quad_dummy_inertia,
                                                           spring_dummy_constant,
                                                           mass_output, tension_forces_output, yaws_output)

    # then we can try something a little less trivial (i.e. move the quads in a circle around obj still stable
