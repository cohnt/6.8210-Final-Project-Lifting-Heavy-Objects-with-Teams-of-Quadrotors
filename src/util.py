import numpy as np
import pydot
from pydrake.all import (
    ExternallyAppliedSpatialForce,
    GeometrySet,
    CollisionFilterDeclaration
)
from IPython.display import display, Image

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

def notebook_plot_plant(plant):
    display(
        Image(
            pydot.graph_from_dot_data(plant.GetTopologyGraphvizString())[0].create_png()
        )
    )

def notebook_plot_diagram(diagram):
    display(
        Image(
            pydot.graph_from_dot_data(diagram.GetGraphvizString())[0].create_png()
        )
    )

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