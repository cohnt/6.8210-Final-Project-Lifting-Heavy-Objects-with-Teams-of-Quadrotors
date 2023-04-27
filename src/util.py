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

def CreateNullExternalForce(plant):
    f = ExternallyAppliedSpatialForce()
    f.body_index = plant.world_body().index()
    return f

def DisableCollisionChecking(sg, context):
    sg_context = sg.GetMyContextFromRoot(context)
    cfm = sg.collision_filter_manager(sg_context)

    query_object = sg.get_query_output_port().Eval(sg_context)
    inspector = query_object.inspector()

    quads = GeometrySet()
    gids = inspector.GetAllGeometryIds()
    for gid in gids:
        # Might want to handle the case where not all geometries are collision geometries?
        quads.Add(gid)
    cfd = CollisionFilterDeclaration()
    cfd.ExcludeWithin(quads)
    cfm.Apply(cfd)

def plot_plant():
    pass # TODO

def plot_diagram():
    pass # TODO

def is_stabilizable(A, B):
    print(A.shape)
    evals, evecs = np.linalg.eig(A)
    for l in evals:
        if np.real(l) >= 0:
            mat = np.hstack((l*np.eye(A.shape[0])-A, B))
            if np.linalg.matrix_rank(mat) != A.shape[0]:
                return False
    return True

def is_detectable(A, C):
    return is_stabilizable(C.T, A.T)