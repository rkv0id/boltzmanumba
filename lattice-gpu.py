import time as tm
import numpy as np
from math import sin, pi
from numba import cuda, float64
from matplotlib.pyplot import imsave

# Flow definition
nx, ny      = 420, 180
r           = ny // 9
dims        = cuda.to_device([nx,ny])
uLB         = 0.04
Re          = 150.0
nulb        = uLB*r/Re
omega       = 1 / (3*nulb+0.5)
consts      = cuda.to_device([uLB, omega])

v = cuda.to_device(
    np.array([
        [1,1], [1,0], [1,-1],
        [0,1], [0,0], [0,-1],
        [-1,1], [-1,0], [-1,-1]
]))
t = cuda.to_device(
    np.array([
        1/36, 1/9, 1/36,
        1/9, 4/9, 1/9,
        1/36, 1/9, 1/36
]))
col1 = cuda.to_device(np.array([0, 1, 2]))
col2 = cuda.to_device(np.array([3, 4, 5]))
col3 = cuda.to_device(np.array([6, 7, 8]))

# CUDA setup
threadsperblock = (16, 8)
blockspergrid_x = int(np.ceil(nx / threadsperblock[0]))
blockspergrid_y = int(np.ceil(ny / threadsperblock[1]))
blockspergrid   = (blockspergrid_x, blockspergrid_y)

@cuda.jit
def initvel(out, uLB, nx, ny):
    x, y = cuda.grid(2)
    if x < nx and y < ny:
        out[0,x,y] = uLB * (1 + 1e-4*sin(y/(ny-1)*2*pi))
        out[1,x,y] = 0

@cuda.jit
def equilibrium(out, rho, u, v, t, nx, ny):
    x, y = cuda.grid(2)
    if x < nx and y < ny:
        u0, u1 = u[0,x,y], u[1,x,y]
        for i in range(9):
            cu = 3 * (v[i,0] * u0 + v[i,1] * u1)
            out[i,x,y] = rho[x,y] * t[i] * \
                (1 + cu + 0.5*cu*cu - 3/2 * (u0*u0 + u1*u1))

@cuda.jit
def post_equilibrium(fin, feq, ny):
    x, y = cuda.grid(2)
    if x == 0 and y < ny:
        for i in range(3):
            fin[i,x,y] = feq[i,x,y] + fin[8-i,x,y] - feq[8-i,x,y]

@cuda.jit
def rho_clc(out, fin, nx, ny):
    x, y = cuda.grid(2)
    if x < nx and y < ny:
        temp = 0
        for i in range(9):
            temp += fin[i,x,y]
        out[x,y] = temp

@cuda.jit
def macroscopic(out, fin, rho, v, nx, ny):
    x, y = cuda.grid(2)
    if x < nx and y < ny:
        temp1, temp2 = 0, 0
        for i in range(9):
            temp1 += v[i,0] * fin[i,x,y]
            temp2 += v[i,1] * fin[i,x,y]
        out[0,x,y] = temp1 / rho[x,y]
        out[1,x,y] = temp2 / rho[x,y]

@cuda.jit
def outflow(fin, col3, nx, ny):
    x, y = cuda.grid(2)
    if x == nx-1 and y < ny:
        for i in col3:
            fin[i,x,y] = fin[i,x-1,y]

@cuda.jit
def inflow(u, rho, vel, fin, col2, col3, ny):
    x, y = cuda.grid(2)
    if x == 0 and y < ny:
        u[0,x,y] = vel[0,x,y]
        u[1,x,y] = vel[1,x,y]
        temp = 0
        for c1,c2 in zip(col2,col3):
            temp += fin[c1,x,y] + 2*fin[c2,x,y]
        rho[x,y] = temp / (1-vel[0,x,y])

@cuda.jit
def collision(out, fin, feq, omega, nx, ny):
    x, y = cuda.grid(2)
    if x < nx and y < ny:
        for i in range(9):
            out[i,x,y] = fin[i,x,y] - omega * (fin[i,x,y] - feq[i,x,y])

@cuda.jit
def bounce(fout, fin, nx, ny):
    x, y = cuda.grid(2)
    if x < nx and y < ny:
        cx  = nx // 4
        cy  = ny // 2
        r   = ny // 9
        if ((x-cx)*(x-cx) + (y-cy)*(y-cy)) < (r*r):
            for i in range(9):
                fout[i,x,y] = fin[8-i,x,y]

@cuda.jit
def stream(fin, fout, v, nx, ny):
    x, y = cuda.grid(2)
    if x < nx and y < ny:
        for i in range(9):
            fin[i,x,y] = fout[i, (x-v[i,0]) % nx, (y-v[i,1]) % ny]

def main(maxIter, stateSv):
    print("Initializing Simulation...")
    fin     = cuda.device_array((9,nx,ny))
    feq     = cuda.device_array((9,nx,ny))
    fout    = cuda.device_array((9,nx,ny))
    rho     = cuda.to_device(np.ones((nx,ny)))
    vel     = cuda.device_array((2,nx,ny))
    u       = cuda.device_array((2,nx,ny))

    # init velocity field
    initvel[blockspergrid, threadsperblock](vel, consts[0], dims[0], dims[1])
    # init equilibrium
    equilibrium[blockspergrid, threadsperblock](fin, rho, vel, v, t, dims[0], dims[1])

    print("Starting Simulation...")
    figures = {}
    start = tm.time()
    for time in range(maxIter):
        # outflow conditions
        outflow[blockspergrid, threadsperblock](fin, col3, dims[0], dims[1])
        
        # new Rho val
        rho_clc[blockspergrid, threadsperblock](rho, fin, dims[0], dims[1])
        
        # macro vel/density
        macroscopic[blockspergrid, threadsperblock](u, fin, rho, v, dims[0], dims[1])
        
        # inflow conditions
        inflow[blockspergrid, threadsperblock](u, rho, vel, fin, col2, col3, dims[1])
        
        # equilibrium state re-compute
        equilibrium[blockspergrid, threadsperblock](feq, rho, vel, v, t, dims[0], dims[1])
        post_equilibrium[blockspergrid, threadsperblock](fin, feq, dims[1])
        
        # collision compute
        collision[blockspergrid, threadsperblock](fout, fin, feq, consts[1], dims[0], dims[1])
        
        # obstacle bounce-back
        bounce[blockspergrid, threadsperblock](fout, fin, dims[0], dims[1])
        
        # streaming to next iter
        stream[blockspergrid, threadsperblock](fin, fout, v, dims[0], dims[1])
        
        # Recording timestamp velocity
        if (time % stateSv == 0):
            fluid = u.copy_to_host()
            fig = np.sqrt(fluid[0] ** 2 + fluid[1] ** 2).transpose() 
            figures[time//stateSv] = fig
    
    end = tm.time()
    print("Ended in %d seconds." % (end - start))
    print("Saving simulation's visuals...")
    for inst, fig in figures.items():
        imsave("gpu_out/vel.{0:04d}.png".format(inst), fig, cmap="autumn")
    print("DONE! Check ./gpu_out folder for progress visuals.")

if __name__ == "__main__":
    main(2000, 500)
