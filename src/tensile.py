import numpy as np
import pydrake.symbolic, pydrake.geometry
from pydrake.all import (
    TemplateSystem,
    LeafSystem_,
    SpatialForce_,
    AbstractValue,
    ExternallyAppliedSpatialForce_,
    AutoDiffXd
)
from pydrake.examples import (
    QuadrotorGeometry
)

@TemplateSystem.define("TensileForce_")
def TensileForce_(T):
    class Impl(LeafSystem_[T]):
        def _construct(self, length, hooke_K, quad_body_index, mass_body_index, converter=None, meshcat=None):
            LeafSystem_[T].__init__(self, converter)
            self.meshcat = meshcat
            self.length = length  # In meters, > 0
            self.hooke_K = hooke_K  # Hooke's law spring constant, > 0
            self.quad_body_index = quad_body_index  # Index of the affected body from the plant
            self.mass_body_index = mass_body_index  # Index of the affected body from the plant

            self.quad_state_input = self.DeclareVectorInputPort("quad_state_input", size=12)
            self.mass_state_input = self.DeclareVectorInputPort("mass_state_input", size=13)  # this is xyz quat state
            self.quad_force_output = self.DeclareAbstractOutputPort("quad_force_output",
                                                                    alloc=lambda: AbstractValue.Make(
                                                                        [ExternallyAppliedSpatialForce_[T]()]),
                                                                    calc=self.QuadOutputForce)
            self.mass_force_output = self.DeclareAbstractOutputPort("mass_force_output",
                                                                    alloc=lambda: AbstractValue.Make(
                                                                        [ExternallyAppliedSpatialForce_[T]()]),
                                                                    calc=self.MassOutputForce)

        def compute_spring_force(self, quad_pos, mass_pos):
            dist = np.linalg.norm(quad_pos - mass_pos)

            if isinstance(quad_pos[0], AutoDiffXd):
                f_mag = (self.hooke_K * (dist - self.length)).max(0)
            else:
                f_mag = pydrake.symbolic.max(0, self.hooke_K * (dist - self.length))
                if len(f_mag.GetVariables()) == 0:
                    f_mag = f_mag.Evaluate()
            f_dir = (mass_pos - quad_pos) / dist
            return f_mag, f_dir

        def QuadOutputForce(self, context, output):
            quad_state = self.quad_state_input.Eval(context)
            quad_pos = quad_state[0:3]

            mass_state = self.mass_state_input.Eval(context)
            mass_pos = mass_state[4:7]

            f_mag, f_dir = self.compute_spring_force(quad_pos, mass_pos)
            f = f_mag * f_dir

            F_Bq_W = SpatialForce_[T](np.zeros((3, 1)), f.reshape(-1, 1))
            p_BoBq_B = np.zeros(3)  # Assume the force is applied at the body origin

            o = ExternallyAppliedSpatialForce_[T]()
            o.body_index = self.quad_body_index
            o.F_Bq_W = F_Bq_W
            o.p_BoBq_B = p_BoBq_B

            output.set_value([o])

            # TODO: Move this to its own system, or include a flag on whether to render or not?
            if type(quad_pos[0]) == np.float64:
                if f_mag == 0:
                    rgba = pydrake.geometry.Rgba(r=1, g=0, b=0)
                else:
                    rgba = pydrake.geometry.Rgba(r=0, g=1, b=0)
                if self.meshcat is not None:
                    self.meshcat.SetLine("cables/cable%d" % self.quad_body_index,
                                         np.array([mass_pos, quad_pos]).T, rgba=rgba)

        def MassOutputForce(self, context, output):
            quad_state = self.quad_state_input.Eval(context)
            quad_pos = quad_state[0:3]

            mass_state = self.mass_state_input.Eval(context)
            mass_pos = mass_state[4:7] # TODO: we may be pulling the wrong state here.

            f_mag, f_dir = self.compute_spring_force(quad_pos, mass_pos)
            f = -f_mag * f_dir  # inverting for mass

            F_Bq_W = SpatialForce_[T](np.zeros((3, 1)), f.reshape(-1, 1))
            p_BoBq_B = np.zeros(3)  # Assume the force is applied at the body origin

            o = ExternallyAppliedSpatialForce_[T]()
            o.body_index = self.mass_body_index
            o.F_Bq_W = F_Bq_W
            o.p_BoBq_B = p_BoBq_B

            output.set_value([o])

        def _construct_copy(self, other, converter=None, ):
            Impl._construct(self, other.length, other.hooke_K,
                            other.quad_body_index, other.mass_body_index, converter=converter)

    return Impl


# Thanks David!
# https://stackoverflow.com/a/72121171/9796174
@TemplateSystem.define("SpatialForceConcatinator_")
def SpatialForceConcatinator_(T):
    class Impl(LeafSystem_[T]):
        def _construct(self, N_inputs, converter=None):
            LeafSystem_[T].__init__(self, converter)
            self.N_inputs = N_inputs
            self.Input_ports = [self.DeclareAbstractInputPort(f"Spatial_Force_{i}",
                                                              AbstractValue.Make([ExternallyAppliedSpatialForce_[T]()]))
                                for i in range(N_inputs)]

            self.Output_port = self.DeclareAbstractOutputPort("Spatial_Forces",
                                                              lambda: AbstractValue.Make(
                                                                  [ExternallyAppliedSpatialForce_[T]()
                                                                   for i in range(N_inputs)]),
                                                              self.Concatenate)

        def Concatenate(self, context, output):
            out = []
            for port in self.Input_ports:
                out += port.Eval(context)
            output.set_value(out)

        def _construct_copy(self, other, converter=None, ):
            Impl._construct(self, other.N_inputs, converter=converter)

    return Impl


@TemplateSystem.define("SpatialForceAdder_")
def SpatialForceAdder_(T):
    class Impl(LeafSystem_[T]):
        def _construct(self, N_inputs, mass_body_index, converter=None):
            LeafSystem_[T].__init__(self, converter)
            self.N_inputs = N_inputs
            self.mass_body_index = mass_body_index
            self.Input_ports = [self.DeclareAbstractInputPort(f"Spatial_Force_{i}",
                                                              AbstractValue.Make([ExternallyAppliedSpatialForce_[T]()]))
                                for i in range(N_inputs)]

            self.Output_port = self.DeclareAbstractOutputPort("force_output",
                                                               alloc=lambda: AbstractValue.Make(
                                                                   [ExternallyAppliedSpatialForce_[T]()]),
                                                               calc=self.AddForces)

        def AddForces(self, context, output):
            f_net = SpatialForce_[T](np.zeros((3, 1)), np.zeros((3, 1)))
            for port in self.Input_ports:
                f_net += port.Eval(context)[0].F_Bq_W

            o = ExternallyAppliedSpatialForce_[T]()
            o.body_index = self.mass_body_index
            o.F_Bq_W = f_net
            o.p_BoBq_B = np.zeros(3)
            output.set_value([o])

        def _construct_copy(self, other, converter=None, ):
            Impl._construct(self, other.N_inputs, other.mass_body_index, converter=converter)

    return Impl


# Default instantations
TensileForce = TensileForce_[None]
SpatialForceConcatinator = SpatialForceConcatinator_[None]
SpatialForceAdder = SpatialForceAdder_[None]