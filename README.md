Use an underactuated-compatible virtual environment. See: https://underactuated.csail.mit.edu/drake.html#section3


Notes for free-body modelling:

Adding a rigid body: https://drake.mit.edu/doxygen_cxx/classdrake_1_1multibody_1_1_multibody_plant.html#a461a59b672f8c3f7c0dcc5caed56d245
Will use the default model instance (index 1)

Work out the inertia matrix of a point (likely diagonal, and degenerate since orientation doesn't matter... 
but that may mess up computations) (can also be created using SpatialInertia.PointMass() static method)

Looks like SpatialInertia.PointMass isn't available from python, so we will construct the spatial inertia
matrix ourselves.

Registering geometry with the SceneGraph: 
1) Call RegisterAsSourceForSceneGraph() (this may already be done on instantiation of MultibodyPlant)
2) Call RegisterVisualGeometry(). Drake has geometric primitives built in: https://drake.mit.edu/doxygen_cxx/classdrake_1_1geometry_1_1_sphere.html#ad4eb22638a3841217373d2caee0d6a74

Then in TensileForce - modify to take in two bodies and output two forces (one on each body) - only need modify the input forces. Make sure the lists are aligned.
