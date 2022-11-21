import numpy as np
import do_mpc
from casadi import *
from casadi.tools import *
import gtsam
from typing import Optional, List
from functools import partial

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
    Kp = 5*6
    Kd = 2*4
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

    for _ in range(maxiter):
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
    # u = -K[:,[0,2]] @ (x[[0,2]]-goal) #this fails miserably
    K = K * np.array([4,1,4,1]) #this is a hack to make it work
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
        'n_horizon': 20,
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

#-------------------------------------------------------------------------------------------------------------
#------------------------------------------------ Factor Graph FULL-------------------------------------------
#-------------------------------------------------------------------------------------------------------------
class factor_graph_full_control():
    def __init__(self, A, B, f_cov):
        self.A : np.ndarray = A
        self.B : np.ndarray = B
        self.f_cov = gtsam.noiseModel.Gaussian.Covariance(f_cov)
        self.h_cov = gtsam.noiseModel.Unit.Create(2) #unary factor
        self.N = 20 #horzinon length
    def make_step(self, x0, cov0, goal):
        #define factors
        graph = gtsam.NonlinearFactorGraph()
        for t in range(self.N):
            x_t = gtsam.symbol('x', t) 
            x_tp1 = gtsam.symbol('x', t + 1)
            u_t = gtsam.symbol('u', t)

            f_factor = gtsam.CustomFactor(self.f_cov, [x_t, x_tp1, u_t], 
                                            partial(error_f, self.A, self.B))
            h_factor = gtsam.CustomFactor(self.h_cov, [x_tp1], 
                                            partial(error_h, goal))
            graph.push_back(f_factor)
            graph.push_back(h_factor)
        prior_factor = gtsam.CustomFactor(gtsam.noiseModel.Gaussian.Covariance(cov0),
                                            [gtsam.symbol('x', 0)],
                                            partial(error_prior, x0))
        graph.push_back(prior_factor)

        #define initial estimates
        initial_estimate = gtsam.Values()
        initial_estimate.insert(gtsam.symbol('x', 0), x0)
        xt = x0
        for t in range(self.N):
            ut = pid_control(xt, goal)
            xtp1 = self.A @ xt + self.B @ ut
            initial_estimate.insert(gtsam.symbol('u', t), ut)
            initial_estimate.insert(gtsam.symbol('x', t+1), xtp1)

        #solve
        params = gtsam.GaussNewtonParams()
        optimizer = gtsam.GaussNewtonOptimizer(graph, initial_estimate, params)
        # params = gtsam.LevenbergMarquardtParams()
        # optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
        result = optimizer.optimize()
        return result.atVector(gtsam.symbol('u',0))


#-------------------------------------------------------------------------------------------------------------
#------------------------------------------------ Factor Graph -----------------------------------------------
#-------------------------------------------------------------------------------------------------------------

# class factor_graph_control():
#     def __init__(self, goal, A, B, DT, R):
#         self.goal : np.ndarray = goal
#         self.A : np.ndarray = A
#         self.B : np.ndarray = B
#         self.DT : np.ndarray = DT
#         self.graph = gtsam.NonlinearFactorGraph()
#         self.initialEstimate = gtsam.Values()
#         self.isam = gtsam.ISAM2()
#         self.R = gtsam.noiseModel.Gaussian.Covariance(R)
#         self.t : int = 0 

#     def control(self, x0, u0, goal):
#         #motion factor
#         x_t = gtsam.symbol('x', self.t) 
#         x_tp1 = gtsam.symbol('x', self.t + 1)
#         u_t = gtsam.symbol('u', self.t)
        
#         x_factor = gtsam.CustomFactor(self.R, [x_t, x_tp1, u_t], partial(error_x, self.A, self.B, self.DT))
#         y_factor = gtsam.CustomFactor(self.R, [x_tp1], partial(error_y, goal))
#         self.graph.push_back(x_factor)
#         self.graph.push_back(y_factor)

#         x_t_est = self.isam2.calculateEstimateV4(x_t)
#         x_tp1_est = self.A @ x_t_est + self.B @ u0
        
#         self.initial_estimate.insert(u_t, u0)
#         self.initial_estimate.insert(x_tp1, x_tp1_est)

#         self.isam.update(self.graph, self.initial_estimate)

#-------------------------------------------------------------------------------------------------------------
#------------------------------------------------ CUSTOM FACTORS ---------------------------------------------
#-------------------------------------------------------------------------------------------------------------
def error_prior(x0, this: gtsam.CustomFactor,
              values: gtsam.Values,
              jacobians: Optional[List[np.ndarray]]) -> float:
    key = this.keys()[0]
    x_estimate = values.atVector(key)
    error = -(x_estimate - x0)
    if jacobians is not None:
        jacobians[0] = np.eye(4)
    return error

def error_h(goal: np.ndarray, this: gtsam.CustomFactor,
              values: gtsam.Values,
              jacobians: Optional[List[np.ndarray]]) -> float:
    key = this.keys()[0]
    x_estimate = values.atVector(key)
    error = -(x_estimate[[0,2]] - goal)
    if jacobians is not None:
        jacobians[0] = np.array([[1, 0, 0, 0],
                                 [0 ,0, 1 ,0]])

    return error

def error_f(A, B, this: gtsam.CustomFactor,
                values: gtsam.Values,
                jacobians: Optional[List[np.ndarray]]) -> float:
        x1_key = this.keys()[0]
        x2_key = this.keys()[1]
        u_key = this.keys()[2]
        estimate_x2 = A @ values.atVector(x1_key) + B @ values.atVector(u_key)
        error = -(estimate_x2 - values.atVector(x2_key))
        if jacobians is not None:
            jacobians[0] = A
            jacobians[1] = np.linalg.inv(A)
            jacobians[2] = B
    
        return error