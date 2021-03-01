
# Flow definition
# Global vars ae compile-time
# constants for Numba's JIT
stateSv     = 500
maxIter     = 200000
Re          = 150.0
nx, ny      = 420, 180
ly          = ny-1
cx, cy, r   = nx//4, ny//2, ny//9
uLB         = 0.04
nulb        = uLB*r/Re
omega       = 1 / (3*nulb+0.5)

col1 = np.array([0, 1, 2])
col2 = np.array([3, 4, 5])
col3 = np.array([6, 7, 8])

v = np.array([
    [1,1], [1,0], [1,-1],
    [0,1], [0,0], [0,-1],
    [-1,1], [-1,0], [-1,-1]
])
t = np.array([
    1/36, 1/9, 1/36,
    1/9, 4/9, 1/9,
    1/36, 1/9, 1/36
])
