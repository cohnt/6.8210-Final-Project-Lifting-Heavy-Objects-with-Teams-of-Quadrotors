import numpy as np
from pydrake.all import (
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Parser,
    SpatialInertia,
    UnitInertia,
    RigidTransform,
    Sphere,
    PropellerInfo,
    Propeller,
    MeshcatVisualizer
)

from underactuated.scenarios import AddFloatingRpyJoint

from tensile import TensileForce, SpatialForceConcatinator

def make_n_quadrotor_system(meshcat, n, cable_length, cable_hooke_K, free_body_mass):
    builder = DiagramBuilder()
    # The MultibodyPlant handles f=ma, but doesn't know about propellers.
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    parser.SetAutoRenaming(True)
    quadrotor_model_instances = []
    for i in range(n):
        (model_instance,) = parser.AddModelsFromUrl(
            "package://drake/examples/quadrotor/quadrotor.urdf"
        )
        quadrotor_model_instances.append(model_instance)
        # By default the multibody has a quaternion floating base.  To match
        # QuadrotorPlant, we can manually add a FloatingRollPitchYaw joint. We set
        # `use_ball_rpy` to false because the BallRpyJoint uses angular velocities
        # instead of ṙ, ṗ, ẏ.
        AddFloatingRpyJoint(
            plant,
            plant.GetFrameByName("base_link", model_instance),
            model_instance,
            use_ball_rpy=False,
        )

    # form spatial inertia of floating mass (SpatialInertia factory methods are not available from Python,
    # so we'll need to construct it ourselves)
    point_inertia = SpatialInertia(free_body_mass, np.zeros(3), UnitInertia.SolidSphere(1))
    # print(point_inertia.IsPhysicallyValid())

    # create floating mass and register visual geometry with scenegraph so it renders
    floating_mass_model_instance = plant.AddModelInstance("floating_mass")
    floating_mass = plant.AddRigidBody("base_link", floating_mass_model_instance, point_inertia)
    plant.RegisterVisualGeometry(
        floating_mass,
        RigidTransform.Identity(),
        Sphere(0.1),
        "floating_mass_sphere",
        np.array([1.0, 0.0, 0.0, 1.0])
    )

    plant.Finalize()

    # Default parameters from quadrotor_plant.cc:
    L = 0.15  # Length of the arms (m).
    kF = 1.0  # Force input constant.
    kM = 0.0245  # Moment input constant.

    # Now we can add in propellers as an external force on the MultibodyPlant.
    prop_info = []
    tensile_forces = []

    mass_body_index =  plant.GetBodyByName("base_link", floating_mass_model_instance).index()
    for model_instance in quadrotor_model_instances:
        quad_body_index = plant.GetBodyByName("base_link", model_instance).index()
        # Note: Rotors 0 and 2 rotate one way and rotors 1 and 3 rotate the other.
        prop_info += [
            PropellerInfo(quad_body_index, RigidTransform([L, 0, 0]), kF, kM),
            PropellerInfo(quad_body_index, RigidTransform([0, L, 0]), kF, -kM),
            PropellerInfo(quad_body_index, RigidTransform([-L, 0, 0]), kF, kM),
            PropellerInfo(quad_body_index, RigidTransform([0, -L, 0]), kF, -kM),
        ]
        tensile_force = builder.AddSystem(TensileForce(cable_length, cable_hooke_K, quad_body_index, mass_body_index, meshcat=meshcat))
        builder.Connect(
            plant.get_state_output_port(model_instance),
            tensile_force.quad_state_input
        )
        builder.Connect(
            plant.get_state_output_port(floating_mass_model_instance),
            tensile_force.mass_state_input
        )
        tensile_forces.append(tensile_force)

    propellers = builder.AddNamedSystem("propeller", Propeller(prop_info))

    combiner = builder.AddNamedSystem("combiner", SpatialForceConcatinator(2))
    builder.Connect(
        propellers.get_output_port(),
        combiner.Input_ports[0]
    )
    builder.Connect(
        combiner.Output_port,
        plant.get_applied_spatial_force_input_port()
    )

    tensile_combiner = builder.AddNamedSystem("tensile_combiner", SpatialForceConcatinator(2 * n))
    # floating_mass_force_adder = builder.AddNamedSystem("floating_mass_force_adder", SpatialForceAdder(n, mass_body_index))
    for i, tensile_force in enumerate(tensile_forces):
        builder.Connect(
            tensile_force.quad_force_output,
            tensile_combiner.Input_ports[i]
        )

        builder.Connect(
            tensile_force.mass_force_output,
            tensile_combiner.Input_ports[n + i]
        )

    builder.Connect(
        tensile_combiner.Output_port,
        combiner.Input_ports[1]
    )

    builder.Connect(
        plant.get_body_poses_output_port(),
        propellers.get_body_poses_input_port(),
    )
    builder.ExportInput(propellers.get_command_input_port(), "u")
    builder.ExportOutput(plant.get_state_output_port(), "q")

    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    return builder.Build(), plant