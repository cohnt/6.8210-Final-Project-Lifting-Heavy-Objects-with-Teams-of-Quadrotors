import numpy as np
from pydrake.all import (
    MathematicalProgram,
    Solve,
    Linearize,
    LinearQuadraticRegulator,
    MakeFiniteHorizonLinearQuadraticRegulator,
    FiniteHorizonLinearQuadraticRegulatorOptions,
    DiagramBuilder,
    Saturation,
    SnoptSolver
)

from util import DisableCollisionChecking, is_stabilizable, is_detectable

def norm_at_least_constraint(vec, dist):
    return np.dot(vec, vec) >= dist**2

def find_fixed_point_snopt(diagram, limit=None, min_quadrotor_distance=None, min_cable_length=None):
    # Given a diagram with a single input and single output port, find a fixed point
    # and a control signal which keeps it at the fixed point.
    n_inputs = diagram.get_input_port(0).size()
    n_outputs = diagram.get_output_port(0).size()

    for i in range(100):
        prog = MathematicalProgram()
        u = prog.NewContinuousVariables(n_inputs, "u")
        x = prog.NewContinuousVariables(n_outputs, "x")

        # for i in range(int(n_inputs/4)):
        #     start = i*6
        #     stop = start + 3
        #     prog.AddConstraint(np.linalg.norm(x[start:stop]) >= 2.5)

        n_quadrotors = int(n_inputs / 4)

        if min_cable_length is not None:
            for i in range(n_quadrotors):
                free_body_start_idx = 6*n_quadrotors + 4
                free_body_pos = x[free_body_start_idx:free_body_start_idx+3]
                quadrotor_start_idx = 6*i
                quadrotor_pos = x[quadrotor_start_idx:quadrotor_start_idx+3]
                offset = free_body_pos - quadrotor_pos
                prog.AddConstraint(norm_at_least_constraint(offset, min_cable_length))

        if min_quadrotor_distance is not None:
            for i in range(int(n_inputs/4)):
                for j in range(i+1, int(n_inputs/4)):
                    start1 = i*6
                    stop1 = start1 + 3
                    start2 = j*6
                    stop2 = start2 + 3
                    prog.AddConstraint(norm_at_least_constraint(x[start1:stop1] - x[start2:stop2], min_quadrotor_distance))

        if limit is not None:
            prog.AddLinearConstraint(u, -limit*np.ones(u.shape), limit*np.ones(u.shape))

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

        x_init = np.random.uniform(low=-5, high=5, size=n_outputs)
        for i in range(n_quadrotors):
            vel_start = 6*i + 3
            x_init[vel_start:vel_start+3] = 0
        prog.SetInitialGuess(x, x_init)
        solver = SnoptSolver()
        # prog.SetSolverOption(solver.solver_id(), "Print file", "snopt_log.txt")
        try:
            result = solver.Solve(prog)
            if result.is_success():
                print(result.is_success(), flush=True)
                break
            else:
                print("Failed")
        except Exception as e:
            print(e, flush=True)
            pass

    return result.GetSolution(x), result.GetSolution(u)

def lqr_stabilize_to_point(system_diagram, fixed_point, fixed_control_signal, Q, R, controller_time_horizon):
    # Constructs an LQR controller for a given system, about a given fixed point with
    # fixed control signal, using the matrices Q and R

    context = system_diagram.CreateDefaultContext()
    context.SetContinuousState(fixed_point)
    sg = system_diagram.GetSubsystemByName("scene_graph")
    DisableCollisionChecking(sg, context)
    system_diagram.get_input_port(0).FixValue(context, fixed_control_signal)

    # linearization = Linearize(system_diagram, context)
    # flag = False
    # if not is_stabilizable(linearization.A(), linearization.B()):
    #     flag = True
    #     print("Warning: (A, B) is not stabilizable! LQR may not work!", flush=True)
    # if not is_detectable(Q, linearization.A()):
    #     flag = True
    #     print("Warning: (Q, A) is not detectable! LQR may not work!", flush=True)

    # if flag:
    #     np.savetxt("A.txt", linearization.A())
    #     np.savetxt("B.txt", linearization.B())
    #     np.savetxt("Q.txt", Q)
    #     np.savetxt("R.txt", R)

    t0 = 0
    tf = controller_time_horizon

    options = FiniteHorizonLinearQuadraticRegulatorOptions()
    options.Qf = Q.copy()

    return MakeFiniteHorizonLinearQuadraticRegulator(system_diagram, context, t0, tf, Q, R, options)

def finite_horizon_lqr_stabilize_to_trajectory(system_diagram, nominal_state_trajectory, nominal_control_trajectory, Q, R):
    # Constructs a finite horizon LQR controller for a given system, about a given nominal
    # trajectory, using the matrices Q and R

    context = system_diagram.CreateDefaultContext()
    sg = system_diagram.GetSubsystemByName("scene_graph")
    DisableCollisionChecking(sg, context)

    t0 = 0
    tf = nominal_state_trajectory.end_time()

    options = FiniteHorizonLinearQuadraticRegulatorOptions()
    options.Qf = Q.copy()
    options.x0 = nominal_state_trajectory
    options.u0 = nominal_control_trajectory

    return MakeFiniteHorizonLinearQuadraticRegulator(system_diagram, context, t0, tf, Q, R, options)

def mpc_stabilize_to_trajectory():
    pass # TODO

def add_controller_to_system(system_diagram, controller, limit=None):
    # Given a system diagram and a controller for that system, returns a bigger diagram
    # in which the controller controls the system.
    builder = DiagramBuilder()
    system_plant = builder.AddNamedSystem("inner_diagram", system_diagram)
    system_controller = builder.AddNamedSystem("controller", controller)
    if limit is None:
        builder.Connect(controller.get_output_port(0), system_plant.get_input_port(0))
        builder.Connect(system_plant.get_output_port(), controller.get_input_port(0))
    else:
        n = controller.get_output_port(0).size()
        min_value = -limit * np.ones(n)
        max_value = -1 * min_value
        saturation = builder.AddNamedSystem("input_limits", Saturation(min_value, max_value))
        builder.Connect(system_plant.get_output_port(), controller.get_input_port(0))
        builder.Connect(controller.get_output_port(0), saturation.get_input_port(0))
        builder.Connect(saturation.get_output_port(0), system_plant.get_input_port(0))

    return builder.Build(), system_plant