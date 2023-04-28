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

# TODO: find proper mass and inertia from the sim or at least programmatically add it in
GRAVITY = np.array([0.0, 0.0, -9.81])


def compute_quad_state_and_control_from_output(sigma, sigma_dt, sigma_ddt, sigma_dddt, sigma_ddddt, mass, inertia):
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

    u_all = np.concatenate([np.array([u_first]), u_rest])
    return sigma[0:3], rpy, sigma_dt[0:3], omega, u_all


def compute_point_mass_state_and_control_from_output(
        x_L_ds,
        tension_dirs_two_to_n_ds,
        yaws_ds,
        load_mass,
        quad_mass,
        quad_inertia
):
    """
    x_L_ds is a 7-tuple, indexed by derivative order
    tension_dirs_two_to_n_ds is a 5-tuple, indexed by derivative order. Elements are 2D arrays,
        first index is quad # 2->n
    yaws_ds is a 5-tuple, indexed by derivative order. Elements are 1D arrays, first index is quad # 1->n
    """

    assert len(x_L_ds) == 7
    assert len(tension_dirs_two_to_n_ds) == 5
    assert len(yaws_ds) == 5

    sum_tension_dirs_two_to_n_ds = np.sum(
        np.hstack(
            tension_dirs_two_to_n_ds
        ), axis=0
    )
    tension_dir_one_ds = load_mass * x_L_ds[2:] + sum_tension_dirs_two_to_n_ds
    tension_dir_one_ds[0] -= load_mass * GRAVITY

    tension_ds_list = [np.hstack([np.expand_dims(first, 0), two_to_n])
                       for first, two_to_n
                       in zip(np.split(tension_dir_one_ds), tension_dirs_two_to_n_ds)]

    # next... compute the unit vectors and tensions
    tension_dirs = tension_ds_list[0]

    # column here to batch mult/div ops and flatten later
    tensions = np.linalg.norm(tension_dirs, axis=1).reshape(-1, 1)
    dirs = tension_dirs / tensions

    tension_dirs_dt = tension_ds_list[1]
    tensions_dt = np.sum(tension_dirs_dt * dirs, axis=1).reshape(-1, 1)
    q_dot = (tension_dirs_dt - tensions_dt * dirs) / tensions

    tension_dirs_ddt = tension_ds_list[2]



def output_map_factory():
    pass  # TODO, returns a function
