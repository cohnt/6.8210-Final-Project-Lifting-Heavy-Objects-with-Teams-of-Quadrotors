

# In[2]:


import sys
sys.path.append("../src")


# In[3]:


import numpy as np
from pydrake.all import (
    StartMeshcat,
    Simulator,
    DirectCollocation,
    Solve,
    PiecewisePolynomial
)


# In[4]:


from world import make_n_quadrotor_system
from util import DisableCollisionChecking
from stabilization import find_fixed_point_snopt, finite_horizon_lqr_stabilize_to_trajectory, add_controller_to_system


# In[5]:


meshcat = StartMeshcat()


# In[6]:


np.random.seed(0)

cable_length = 2
cable_hooke_K = 10
free_body_mass = 1
n_quadrotors = 4
quadrotor_input_limit = 3 # or None

min_quadrotor_distance = 1 # or None
min_cable_length = 2.1
controller_time_horizon = 10


# In[7]:


diagram, plant = make_n_quadrotor_system(meshcat,
                                         n_quadrotors,
                                         cable_length,
                                         cable_hooke_K,
                                         free_body_mass)


# In[8]:


fixed_point, fixed_control = find_fixed_point_snopt(diagram,
                                                    limit=quadrotor_input_limit,
                                                    min_quadrotor_distance=min_quadrotor_distance,
                                                    min_cable_length=min_cable_length)

fixed_point_2 = fixed_point.copy()
xyz = np.array([10, 0, 0])
for i in range(n_quadrotors):
    fixed_point_2[6*i:6*i+3] += xyz
free_body_start = 6*n_quadrotors+4
fixed_point_2[free_body_start:free_body_start+3] += xyz


# In[9]:


n_steps = 100
min_time_step = 1e-5
max_time_step = 1e-1

trajopt = DirectCollocation(diagram, diagram.CreateDefaultContext(), n_steps, min_time_step, max_time_step)
breaks = np.array([0,n_steps*min_time_step])
control_knots = np.tile(fixed_control, (2,1)).T
state_knots = np.vstack((fixed_point, fixed_point_2)).T
u_init = PiecewisePolynomial.ZeroOrderHold(breaks, control_knots)
x_init = PiecewisePolynomial.FirstOrderHold(breaks, state_knots)

trajopt.SetInitialTrajectory(u_init, x_init)

prog = trajopt.prog()

prog.AddLinearEqualityConstraint(np.eye(len(fixed_point)), fixed_point, trajopt.initial_state())
prog.AddLinearEqualityConstraint(np.eye(len(fixed_point)), fixed_point_2, trajopt.final_state())
for i in range(n_steps):
    limit = np.ones(4*n_quadrotors) * quadrotor_input_limit
    prog.AddBoundingBoxConstraint(-limit, limit, trajopt.input(i))
    
result = Solve(prog)
print(result.is_success())


# In[ ]:


input_traj = trajopt.ReconstructInputTrajectory(result)
state_traj = trajopt.ReconstructStateTrajectory(result)

# Make an LQR controller

Q_quadrotor_pos = [10.] * 6
Q_quadrotor_vel = [1.] * 6
Q_freebody_pos = [1.] * 4 + [10.] * 3
Q_freebody_vel = [1.] * 3 + [1.] * 3
Q_pos = Q_quadrotor_pos * n_quadrotors + Q_freebody_pos
Q_vel = Q_quadrotor_vel * n_quadrotors + Q_freebody_vel
Q = np.diag(Q_pos + Q_vel)
R = np.eye(4 * n_quadrotors)

lqr_controller = finite_horizon_lqr_stabilize_to_trajectory(diagram, state_traj, input_traj, Q, R)


# In[ ]:




