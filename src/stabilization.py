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
    Sphere
)
from pydrake.examples import (
    QuadrotorGeometry
)
from IPython.display import display, SVG, Image

from underactuated.scenarios import AddFloatingRpyJoint

from tensile import TensileForce, SpatialForceConcatinator

from pydrake.systems.primitives import Adder

def find_fixed_point_snopt():
	pass # TODO

def lqr_stabilize_to_point(system_diagram, fixed_point, fixed_control_signal, Q, R):
    # Constructs an LQR controller for a given system, about a given fixed point with
    # fixed control signal, using the matrices Q and R
	context = system_diagram.CreateDefaultContext()
    context.SetContinuousState(fixed_point)
    diagram.get_input_port(0).FixValue(fixed_control_signal)
    return LinearQuadraticRegulator(system_diagram, context, Q, R)

def finite_horizon_lqr_stabilize_to_trajectory():
	pass # TODO

def mpc_stabilize_to_trajectory():
	pass # TODO

def add_controller_to_system(system_diagram, controller):
    # Given a system diagram and a controller for that system, returns a bigger diagram
    # in which the controller controls the system.
    builder = DiagramBuilder()
    system_plant = builder.AddNamedSystem('inner_diagram', system_diagram)
    builder.Connect(controller.get_output_port(0), system_plant.get_input_port(0))
    builder.Connect(system_plant.get_output_port(), controller.get_input_port(0))

    return builder.Build(), system_plant