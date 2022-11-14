import symforce.geo as geo
import symforce.symbolic as sf

p = geo.V2.symbolic('p')
b = geo.V2.symbolic('b')

#test range
r = (b - p).norm()
drange_dp = [r.diff(p[0]), r.diff(p[1])]
print(drange_dp)

dy = b[1] - p[1]
dx = b[0] - p[0]
theta = sf.atan2(dy,dx)
dtheta_dp = [theta.diff(p[0]), theta.diff(p[1])]
print(dtheta_dp)