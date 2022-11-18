import numpy as np
from numpy.matlib import repmat
import do_mpc
from casadi import *
from casadi.tools import *

#-------------------------------------------------------------------------------------------------------------
#------------------------------------------------ PID CONTROL ------------------------------------------------
#-------------------------------------------------------------------------------------------------------------

def pid_control(x, goal):
    '''
    x is the state vector [x, vx, y, vy]
    start is the starting point
    goal is the goal point
    '''
    p = x[[0,2]]
    v = x[[1,3]]
    e = goal - p
    de = -v
    Kp = 5
    Kd = 2
    u = Kp * e + Kd * de
    return u

#-------------------------------------------------------------------------------------------------------------
#------------------------------------------------ LQR --------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------

def solve_DARE(A, B, Q, R, maxiter=150, eps=0.01):
    """
    Solve a discrete time_Algebraic Riccati equation (DARE)
    """
    P = Q

    for i in range(maxiter):
        Pn = A.T @ P @ A - A.T @ P @ B @ \
            np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A + Q
        if (abs(Pn - P)).max() < eps:
            break
        P = Pn

    return Pn


def dlqr(A, B, Q, R):
    """
    Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    # ref Bertsekas, p.151
    """

    # first, try to solve the ricatti equation
    P = solve_DARE(A, B, Q, R)

    # compute the LQR gain
    K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)

    eigVals, eigVecs = np.linalg.eig(A - B @ K)
    return K, P, eigVals


def lqr_control(x, goal, A, B, Q, R):
    goal_full = np.array([goal[0],0.0,goal[1],0.0])
    K, _, _ = dlqr(A, B, Q, R)
    u = -K @ (x - goal_full)
    return u

#-------------------------------------------------------------------------------------------------------------
#------------------------------------------------ MPC --------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------

def build_mpc(goal, A, B, DT):
    symvar_type = 'SX'
    model_type = 'discrete' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)
    
    _x = model.set_variable(var_type='_x', var_name='x', shape=(4,1))
    _u = model.set_variable(var_type='_u', var_name='u', shape=(2,1))
    model.set_expression(expr_name='cost', expr=sum1((goal - _x[[0,2]])**2))

    x_next = A@_x+B@_u
    model.set_rhs('x', x_next)
    model.setup()

    mpc = do_mpc.controller.MPC(model)
    setup_mpc = {
        'n_robust': 0,
        'n_horizon': 100,
        't_step': DT,
        'store_full_solution':False,
    }
    mpc.set_param(**setup_mpc)
    
    mterm = model.aux['cost']
    lterm = model.aux['cost'] # terminal cost

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u=1e-4)
    mpc.setup()
    return mpc

def mpc_control(mpc,x0):
    mpc.x0 = x0
    mpc.set_initial_guess()
    return mpc.make_step(x0)