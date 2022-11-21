import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import control


EPS = 1e-16

def h_gps(x, gps_cov = None):
    p = x[[0,2]]
    if gps_cov is not None:
        z = np.random.multivariate_normal(p, gps_cov)
    else:
        z = p
    H = np.array([[1, 0, 0, 0],
                  [0 ,0, 1 ,0]])
    return z, H

def bearing_(p, b):
    d = b - p
    d2 = d[0]**2 + d[1]**2
    theta = np.arctan2(d[1], d[0])
    H = [d[1]/d2, -d[0]/d2]
    return theta, H

def range_(p ,b):
    '''
    range defined is the distance from p to b
    '''
    d = b - p
    d2 = d[0]**2 + d[1]**2
    r = np.sqrt(d2)
    H = max(1/r,EPS) * np.array([-d[0], -d[1]])
    return r, H

def h_beacons(x, beacons, beacon_cov = None):
    '''
    beacons is a Nx2 array of beacon locations
    beacon_cov is a 2x2 array of covariance ~ [range, bearing]
    '''
    p = x[[0,2]]
    h = np.zeros((beacons.shape[0] * 2))
    H = np.zeros((beacons.shape[0] * 2, 4))
    for i in range(beacons.shape[0]):
        r, dr_dp = range_(p, beacons[i])
        angle, dangle_dp = bearing_(p, beacons[i])
        if beacon_cov is not None:
            h[[2*i,2*i+1]] = np.random.multivariate_normal([r, angle], beacon_cov)
        else:
            h[[2*i,2*i+1]] = [r, angle]

        H[2*i] = np.array([dr_dp[0], 0, dr_dp[1], 0])
        H[2*i+1] = np.array([dangle_dp[0], 0, dangle_dp[1], 0])
    return h, H

def angle_diff(theta1,theta2):
    '''
    return theta1 - theta2 in the range [-pi, pi]
    '''
    z1 = np.exp(theta1 * 1j)
    z2 = np.exp(theta2 * 1j)
    z = z1*np.conjugate(z2)
    return np.angle(z)

def f(x, u, DT, C, M):
    '''
    x is the state vector [x, vx, y, vy]
    u is the control vector [ax, ay]
    '''
    px = x[0]
    vx = x[1]
    py = x[2]
    vy = x[3]

    #compute friction
    v = np.sqrt(vx**2 + vy**2)
    f = C * v
    fx = f * vx / (v + EPS)
    fy = f * vy / (v + EPS)

    ax = (u[0] - fx)/M
    ay = (u[1] - fy)/M

    x  = np.zeros(4)
    x[0] = px + vx * DT + 0.5 * ax * DT**2
    x[1] = vx + ax * DT
    x[2] = py + vy * DT + 0.5 * ay * DT**2
    x[3] = vy + ay * DT
    
    Fx = np.array([[1, DT, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, DT],
                   [0, 0, 0, 1]])
    return x, Fx

def EKF(x_k, cov_k, dt, c, m, u, z_gps, z_beacons, beacons, Q, R):
    z = np.hstack((z_gps, z_beacons))
    
    x_pred, F = f(x_k, u, dt, c, m)
    cov_pred = F @ cov_k @ F.T + Q
    
    z_gps_hat, H_gps = h_gps(x_pred)
    z_beacons_hat, H_beacons = h_beacons(x_pred, beacons)

    H = np.zeros((z.shape[0], x_k.shape[0]))
    H[:2] = H_gps
    H[2:] = H_beacons

    S = H @ cov_pred @ H.T + R
    K = cov_pred @ H.T @ np.linalg.inv(S)

    dz = np.zeros_like(z) #z - h(x_predict)
    dz[:2] = z[:2] - z_gps_hat
    for i in range(z[2:].shape[0]):
        if i % 2 == 0: #range measurement
            dz[2+i] = z[2+i] - z_beacons_hat[i]
        else: #bearing measurement
            dz[2+i] = angle_diff(z[2+i], z_beacons_hat[i])

    x_kp1 = x_pred + K @ dz
    cov_kp1 = (np.eye(4) - K @ H) @ cov_pred
    return x_kp1, cov_kp1

def EKF_gps(x_k, cov_k, dt, c, m, u, z_gps, R, Q):
    z = z_gps

    x_pred, F = f(x_k, u, dt, c, m)
    cov_pred = F @ cov_k @ F.T + R
    
    z_gps_hat, H_gps = h_gps(x_pred)

    H = H_gps
    H[:2] = H_gps

    S = H @ cov_pred @ H.T + Q
    K = cov_pred @ H.T @ np.linalg.inv(S)

    x_kp1 = x_pred + K @ (z - z_gps_hat)
    cov_kp1 = (np.eye(4) - K @ H) @ cov_pred
    return x_kp1, cov_kp1

def x_cov_to_p_cov(cov):
    return [[cov[0,0], cov[0,2]], [cov[2,0], cov[2,2]]]

def plot_pos(pos, cov, nstd=3, ax : plt.Axes = None, facecolor = 'none',edgecolor = 'b' ,  **kwargs):
    eigs, vecs = np.linalg.eig(cov)
    theta = np.degrees(np.arctan2(vecs[1,0],vecs[0,0])) #obtain theta from first axis. second axis is just perpendicular to it

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(eigs)
    ellip = Ellipse(xy=pos, 
                    width=width, 
                    height=height, 
                    angle=theta,
                    facecolor = facecolor, 
                    edgecolor=edgecolor, **kwargs)
    if ax is not None:
            ax.add_patch(ellip)      
    return ellip


def main():
    #CONSTANTS FOR MOTION
    C = 1.0
    M = 1.0
    DT = 0.01

    #TARGETS
    start = np.array([1.0 ,1.0])
    goal = np.array([5.0, 4.0])

    #CONSTANTS FOR ESTIMATION
    gps_cov = 0.1 * np.eye(2)
    beacons_cov = np.diag([0.1, np.radians(3)])
    beacons = np.array([[10.0,5.0],
                        [3.0, 6.0],
                        [8.0, 2.0]])
    R = np.zeros((8,8))
    R[:2,:2] = gps_cov
    R[2:,2:] = np.kron(np.eye(3), beacons_cov)
    Q = np.diag([0.001, 0.1, 0.001, 0.1])*0.1

    #CONSTANTS FOR LQR CONTROL
    Qlqr = np.eye(4) * 200
    Rlqr = np.eye(2)
    Ad = np.array([[1, DT, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, DT],
                    [0, 0, 0, 1]])
    Bd = np.array([[0.5*DT**2, 0],
                    [DT, 0],
                    [0, 0.5*DT**2],
                    [0, DT]])

    #INITALIZATION FOR MPC
    mpc = control.build_mpc(goal, Ad, Bd, DT)

    #INITALIZATION FOR FACTOR GRAPH CONTROL (FULL)
    fgc = control.factor_graph_full_control(Ad, Bd, Q)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(start[0], start[1], marker='o', color='b', s = 30)
    ax.scatter(goal[0], goal[1], marker='o', color='r', s = 30)
    for b in beacons:
        ax.scatter(b[0], b[1], marker = 'd', color = 'b', s = 30)
    
    x_hat = x_hat_prev = x = np.array([start[0], 0.0, start[1], 0.0])
    xcov_hat = xcov_hat_prev = np.diag([0.1, 0.001, 0.1, 0.001])
    u_prev = np.zeros(2)
    
    history = {"x": [], 
               "x_hat" : [], "xcov_hat" : [], 
               "z_gps" : [], "z_beacons" : [],
                "u" : []}
    disk_plot = ax.scatter(x[0], x[2], marker='s', color='g')
    est_plot = plot_pos(x_hat[[0,2]], x_cov_to_p_cov(xcov_hat), ax = ax)
    time_vec = np.arange(0, 0.5, DT)
    
    with plt.ion():
        for t in time_vec:
            #measure
            z_gps, _ = h_gps(x, gps_cov)
            z_beacons, _ = h_beacons(x, beacons, beacons_cov)

            #ESTIMATION
            # x_hat, xcov_hat  = EKF_gps(x_hat_prev, xcov_hat_prev, DT, C, M, u_prev, z_gps, Q, gps_cov)
            x_hat, xcov_hat = EKF(x_hat_prev, xcov_hat_prev, DT, C, M, u_prev, z_gps, z_beacons, beacons, Q, R)
            
            #lame estiamtion
            # phat = z_gps
            # vhat = (phat - x_hat_prev[[0,2]])/DT
            # x_hat = np.array([phat[0], vhat[0], phat[1], vhat[1]])
            
            #CONTROL
            # u = control.pid_control(x_hat, goal)
            # u = control.lqr_control(x_hat, goal, Ad, Bd, Qlqr, Rlqr)
            # u = control.mpc_control(mpc, x_hat)
            u = fgc.make_step(x_hat, xcov_hat ,goal)

            #GROUND TRUTH
            x, _ = f(x, u, DT, C, M)

            #store for next iteration
            x_hat_prev = x_hat
            xcov_hat_prev = xcov_hat
            u_prev = u

            #plot
            disk_plot.set_offsets(x[[0,2]])
            est_plot.remove()
            est_plot = plot_pos(x_hat[[0,2]], x_cov_to_p_cov(xcov_hat), ax = ax)

            #log history
            history["z_gps"].append(z_gps)
            history["z_beacons"].append(z_beacons)
            history["u"].append(u)
            history["x"].append(x)
            history["x_hat"].append(x_hat)
            plt.pause(DT)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(time_vec[[0,-1]], [goal[0], goal[0]], color = 'r')
    ax.plot(time_vec[[0,-1]], [goal[1], goal[1]], color = 'k')
    ax.plot(time_vec, [p[0] for p in history["x"]], color = 'r')
    ax.plot(time_vec, [p[2] for p in history["x"]], color = 'k')
    ax.plot(time_vec, [p[0] for p in history["x_hat"]], color = 'r', linestyle = '--')
    ax.plot(time_vec, [p[2] for p in history["x_hat"]], color = 'k', linestyle = '--')
    ax.scatter(time_vec, [p[0] for p in history["z_gps"]], color = 'r')
    ax.scatter(time_vec, [p[1] for p in history["z_gps"]], color = 'k')
    plt.show()

if __name__ == '__main__':
    main()

