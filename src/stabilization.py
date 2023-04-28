import numpy as np
from pydrake.all import (
    MathematicalProgram,
    Solve,
    Linearize,
    LinearQuadraticRegulator,
    DiagramBuilder
)

from util import DisableCollisionChecking, is_stabilizable, is_detectable

def find_fixed_point_snopt(diagram):
    # Given a diagram with a single input and single output port, find a fixed point
    # and a control signal which keeps it at the fixed point.
    n_inputs = diagram.get_input_port(0).size()
    n_outputs = diagram.get_output_port(0).size()

    for i in range(100):
        prog = MathematicalProgram()
        u = prog.NewContinuousVariables(n_inputs, "u")
        x = prog.NewContinuousVariables(n_outputs, "x")

        prog.AddLinearConstraint(x[-7:-4], np.zeros(3), np.zeros(3))

        for i in range(int(n_inputs/4)):
            start = i*6
            stop = start + 3
            prog.AddConstraint(np.linalg.norm(x[start:stop]) >= 2.5)

        def time_derivative(decision_variables):
            x = decision_variables[:n_outputs]
            u = decision_variables[n_outputs:]
            ad_diagram = diagram.ToAutoDiffXd()
            context = ad_diagram.CreateDefaultContext()
            sg = ad_diagram.GetSubsystemByName("scene_graph")
            DisableCollisionChecking(sg, context)
            ad_diagram.get_input_port(0).FixValue(context, u)
            context.SetContinuousState(x)
            output = ad_diagram.EvalTimeDerivatives(context)
            return output.CopyToVector()

        prog.AddConstraint(time_derivative, np.zeros(n_outputs), np.zeros(n_outputs), vars=np.hstack((x, u)))

        prog.AddQuadraticCost(np.eye(len(x)), np.zeros(len(x)), x)
        prog.AddQuadraticCost(np.eye(len(u)), np.zeros(len(u)), u)

        prog.SetInitialGuess(x, np.random.normal(scale=10, size=n_outputs))
        try:
            result = Solve(prog)
            if result.is_success():
                print(result.is_success())
                break
        except:
            pass

    return result.GetSolution(x), result.GetSolution(u)

def lqr_stabilize_to_point(system_diagram, fixed_point, fixed_control_signal, Q, R):
    # Constructs an LQR controller for a given system, about a given fixed point with
    # fixed control signal, using the matrices Q and R

    context = system_diagram.CreateDefaultContext()
    context.SetContinuousState(fixed_point)
    sg = system_diagram.GetSubsystemByName("scene_graph")
    DisableCollisionChecking(sg, context)
    system_diagram.get_input_port(0).FixValue(context, fixed_control_signal)

    linearization = Linearize(system_diagram, context)
    flag = False
    if not is_stabilizable(linearization.A(), linearization.B()):
        flag = True
        print("Warning: (A, B) is not stabilizable! LQR may not work!", flush=True)
    if not is_detectable(Q, linearization.A()):
        flag = True
        print("Warning: (Q, A) is not detectable! LQR may not work!", flush=True)

    if flag:
        np.savetxt("A.txt", linearization.A())
        np.savetxt("B.txt", linearization.B())
        np.savetxt("Q.txt", Q)
        np.savetxt("R.txt", R)

    return LinearQuadraticRegulator(system_diagram, context, Q, R)

def finite_horizon_lqr_stabilize_to_trajectory():
    pass # TODO

def mpc_stabilize_to_trajectory():
    pass # TODO

def add_controller_to_system(system_diagram, controller):
    # Given a system diagram and a controller for that system, returns a bigger diagram
    # in which the controller controls the system.
    builder = DiagramBuilder()
    system_plant = builder.AddNamedSystem("inner_diagram", system_diagram)
    system_controller = builder.AddNamedSystem("controller", controller)
    builder.Connect(controller.get_output_port(0), system_plant.get_input_port(0))
    builder.Connect(system_plant.get_output_port(), controller.get_input_port(0))

    return builder.Build(), system_plant