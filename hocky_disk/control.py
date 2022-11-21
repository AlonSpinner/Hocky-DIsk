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
        self.u_cov = gtsam.noiseModel.Unit.Create(1) #unary factor
        self.N = 60 #horzinon length
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
            u_factor = gtsam.CustomFactor(self.u_cov, [u_t],
                                            partial(error_u))

            graph.push_back(f_factor)
            graph.push_back(h_factor)
            graph.push_back(u_factor)
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

class factor_graph_incremental_control:
    def __init__(self, A, B, f_cov, x0, cov0, goal):
        self.A : np.ndarray = A
        self.B : np.ndarray = B
        self.f_cov = gtsam.noiseModel.Gaussian.Covariance(f_cov)
        self.h_cov = gtsam.noiseModel.Unit.Create(2) #unary factor
        self.u_cov = gtsam.noiseModel.Unit.Create(1) #unary factor
        self.N = 60 #horzinon length

        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.t = 0

        params = gtsam.ISAM2Params()
        params.setRelinearizeThreshold(0.01)
        params.setRelinearizeSkip(1)
        
        self.isam2 = gtsam.ISAM2(params)

        #initalize system
        for t in range(self.N):
            x_t = gtsam.symbol('x', t) 
            x_tp1 = gtsam.symbol('x', t + 1)
            u_t = gtsam.symbol('u', t)

            f_factor = gtsam.CustomFactor(self.f_cov, [x_t, x_tp1, u_t], 
                                            partial(error_f, self.A, self.B))
            h_factor = gtsam.CustomFactor(self.h_cov, [x_tp1], 
                                            partial(error_h, goal))
            self.graph.push_back(f_factor)
            self.graph.push_back(h_factor)
        prior_factor = gtsam.CustomFactor(gtsam.noiseModel.Gaussian.Covariance(cov0),
                                [gtsam.symbol('x', 0)],
                                partial(error_prior, x0))
        self.graph.push_back(prior_factor)
        self.initial_estimate.insert(gtsam.symbol('x', 0), x0)
        xt = x0
        for t in range(self.N):
            ut = pid_control(xt, goal)
            xtp1 = self.A @ xt + self.B @ ut
            self.initial_estimate.insert(gtsam.symbol('u', t), ut)
            self.initial_estimate.insert(gtsam.symbol('x', t+1), xtp1)
        self.isam2.update(self.graph, self.initial_estimate)
        self.isam2.calculateBestEstimate()
        self.initial_estimate.clear() 

    def make_step(self, xt, covt, goal):
            #define factors
            tpN = self.t + self.N
            x_tpN = gtsam.symbol('x', tpN) 
            x_tpNp1 = gtsam.symbol('x', tpN + 1)
            u_tpN = gtsam.symbol('u', tpN)

            f_factor = gtsam.CustomFactor(self.f_cov, [x_tpN, x_tpNp1, u_tpN], 
                                            partial(error_f, self.A, self.B))
            h_factor = gtsam.CustomFactor(self.h_cov, [x_tpNp1], 
                                            partial(error_h, goal))
            self.graph.push_back(f_factor)
            self.graph.push_back(h_factor)
            
            if self.t != 0:
                prior_factor = gtsam.CustomFactor(gtsam.noiseModel.Gaussian.Covariance(covt),
                                                    [gtsam.symbol('x', self.t)],
                                                    partial(error_prior, xt))
                self.graph.push_back(prior_factor)

            #define initial estimates
            xtpN = self.isam2.calculateEstimateVector(gtsam.symbol('x', tpN))
            utpN = pid_control(xtpN, goal)
            xtpNp1 = self.A @ xtpN + self.B @ utpN
            self.initial_estimate.insert(gtsam.symbol('u', tpN), utpN)
            self.initial_estimate.insert(gtsam.symbol('x', tpN+1), xtpNp1)

            #solve
            self.isam2.update(self.graph, self.initial_estimate)
            self.isam2.calculateBestEstimate()
            self.initial_estimate.clear() 
   
            #increment time
            self.t += 1
            return self.isam2.calculateEstimate(gtsam.symbol('u',self.t))
#-------------------------------------------------------------------------------------------------------------
#------------------------------------------------ CUSTOM FACTORS ---------------------------------------------
#-------------------------------------------------------------------------------------------------------------
def error_prior(x0, this: gtsam.CustomFactor,
              values: gtsam.Values,
              jacobians: Optional[List[np.ndarray]]) -> float:
    key = this.keys()[0]
    x_estimate = values.atVector(key)
    error =  x0 - x_estimate
    if jacobians is not None:
        jacobians[0] = np.eye(4)
    return error

def error_h(goal: np.ndarray, this: gtsam.CustomFactor,
              values: gtsam.Values,
              jacobians: Optional[List[np.ndarray]]) -> float:
    key = this.keys()[0]
    x_estimate = values.atVector(key)
    error = goal - x_estimate[[0,2]]
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
        error = values.atVector(x2_key) - estimate_x2
        if jacobians is not None:
            jacobians[0] = A
            jacobians[1] = np.linalg.inv(A)
            jacobians[2] = B
    
        return error

def error_u(this: gtsam.CustomFactor,
                values: gtsam.Values,
                jacobians: Optional[List[np.ndarray]]) -> float:
        u_key = this.keys()[0]
        u_val = values.atVector(u_key)

        k = 1e-8
        error = k * np.array([u_val[0]**2 + u_val[1]**2])
        if jacobians is not None:
            jacobians[0] = np.reshape(k * 2 * u_val, (1,2))
        return error

        # u_val_x = u_val[0]
        # if u_val_x > 1000.0:
        #     error_x = (u_val_x-1000.0)**2
        #     j_x = 2*(u_val_x-1000.0)
        # elif u_val_x < 1000.0:
        #     error_x = (u_val_x+1000.0)**2
        #     j_x = 2*(u_val_x+1000.0)
        # else:
        #     error_x = 0
        #     j_x = 0

        # u_val_y = u_val[1]
        # if u_val_y > 1000.0:
        #     error_y = (u_val_y-1000.0)**2
        #     j_y = 2*(u_val_y-1000.0)
        # elif u_val_y < 1000.0:
        #     error_y = (u_val_y+1000.0)**2
        #     j_y = 2*(u_val_y+1000.0)
        # else:
        #     error_y = 0
        #     j_y = 0

        # error = np.array([error_x,error_y])
        # if jacobians is not None:
        #     jacobians[0] = np.array([[j_x, 0],
        #                              [0, j_y]])
        # return error