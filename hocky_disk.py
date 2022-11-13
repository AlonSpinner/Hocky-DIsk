import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-16

def h_gps(x, gps_cov):
    p = x[[0,2]]
    z = np.random.multivariate_normal(p, gps_cov)
    H = np.array([[1, 0, 0, 0],
                  [0 ,0, 1 ,0]])
    return z, H

def h_beacons(x, beacons, beacon_cov):
    '''
    beacons is a Nx2 array of beacon locations
    beacon_cov is a 2x2 array of covariance ~ [range, bearing]
    '''
    p = x[[0,2]]
    h = np.zeros((beacons.shape[0], 2))
    Hs = []
    for i in range(beacons.shape[0]):
        r = np.linalg.norm(beacons[i] - p)
        angle = np.arctan2(beacons[i,1] - p[1], beacons[i,0] - p[0])
        h[i] = np.random.multivariate_normal([r, angle], beacon_cov)
    return h

def f(x, u, dt, c, m):
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
    f = c * v
    fx = f * vx / (v + EPS)
    fy = f * vy / (v + EPS)

    ax = (u[0] - fx)/m
    ay = (u[1] - fy)/m

    x  = np.zeros(4)
    x[0] = px + vx * dt + 0.5 * ax * dt**2
    x[1] = vx + ax * dt
    x[2] = py + vy * dt + 0.5 * ay * dt**2
    x[3] = vy + ay * dt
    return x

def control(x, goal):
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

def main():
    c = 1.0
    m = 1.0
    dt = 0.01
    gps_cov = 0.1 * np.eye(2)
    beacons_cov = np.diag([0.1, np.radians(3)])
    beacons = np.array([[10.0,5.0],
                        [3.0, 6.0],
                        [8.0, 2.0]])

    start = np.array([1.0 ,1.0])
    goal = np.array([5.0, 4.0])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(start[0], start[1], marker='o', color='b', s = 30)
    ax.scatter(goal[0], goal[1], marker='o', color='r', s = 30)
    for b in beacons:
        ax.scatter(b[0], b[1], marker = 'd', color = 'b', s = 30)
    xhat_prev = x = np.array([start[0], 0.0, start[1], 0.0])
    history = {"x": [], "x_hat" : [], "z_gps" : [], "z_beacons" : [], "u" : []}
    disk_plot = ax.scatter(x[0], x[2], marker='s', color='g')
    time_vec = np.arange(0, 5, dt)
    with plt.ion():
        for t in time_vec:
            z_gps, H_gps = h_gps(x, gps_cov)
            z_beacons = h_beacons(x, beacons, beacons_cov)

            phat = z_gps
            vhat = (phat - xhat_prev[[0,2]])/dt
            xhat = np.array([phat[0], vhat[0], phat[1], vhat[1]])
            
            u = control(xhat, goal)
            x = f(x, u, dt, c, m)

            xhat_prev = xhat

            disk_plot.set_offsets(x[[0,2]])
            history["z_gps"].append(z_gps)
            history["z_beacons"].append(z_beacons)
            history["u"].append(u)
            history["x"].append(x)
            plt.pause(dt)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(time_vec[[0,-1]], [goal[0], goal[0]], color = 'r')
    ax.plot(time_vec[[0,-1]], [goal[1], goal[1]], color = 'k')
    ax.plot(time_vec, [p[0] for p in history["x"]], color = 'r')
    ax.plot(time_vec, [p[2] for p in history["x"]], color = 'k')
    ax.scatter(time_vec, [p[0] for p in history["z_gps"]], color = 'r')
    ax.scatter(time_vec, [p[1] for p in history["z_gps"]], color = 'k')
    plt.show()


if __name__ == '__main__':
    main()

