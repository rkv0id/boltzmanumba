import time as tm
import numpy as np
from math import sin, pi
from numba import cuda, float64
from matplotlib.pyplot import imsave, imshow

# Flow definition
nx, ny      = 420, 180
ly          = ny-1
cx, cy, r   = nx//4, ny//2, ny//9
uLB         = 0.04
Re          = 150.0
nulb        = uLB*(ny//9)/Re
omega       = 1 / (3*nulb+0.5)

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
col1 = np.array([0, 1, 2])
col2 = np.array([3, 4, 5])
col3 = np.array([6, 7, 8])

# CUDA setup
threadsperblock = (16, 8)
blockspergrid_x = int(np.ceil(nx / threadsperblock[0]))
blockspergrid_y = int(np.ceil(ny / threadsperblock[1]))
blockspergrid   = (blockspergrid_x, blockspergrid_y)

@cuda.jit
def init_simul(rho, feq):
    cuda.syncthreads()
    j, k = cuda.grid(2)
    if j < nx and k < ny :
        # initial velocity field
        u_0 = uLB * (1 + 1e-4*sin(k/ly*2*pi))
        u_1 = 0
        for i in range(9):
            cu = 3 * (v[i,0]*u_0 + v[i,1]*u_1)
            feq[i,j,k] = rho * t[i] * (1 + cu + 0.5*cu**2 - 3/2 * (u_0**2 + u_1**2))

@cuda.jit
def cell_iter(fin, fout, u):
    j, k = cuda.grid(2)
    if j<nx and k<ny:
        feq = cuda.local.array(9, dtype=float64)
        
        # outflow conditions
        if j==(nx-1):
            for i in col3:
                fin[i,j,k] = fin[i,j-1, k] 

        # macroscopic vars compute
        rho = 0    
        u_0 = 0
        u_1 = 0
        for i in range(9):
            rho += fin[i,j,k]    
            u_0 += v[i,0] * fin[i,j,k]
            u_1 += v[i,1] * fin[i,j,k]
        u_0 /= rho
        u_1 /= rho
        
        # inflow conditions
        if j == 0:
            rho = 0
            u_0 = uLB * (1 + 1e-4*sin(k/ly*2*pi))
            u_1 = 0
            for i_1, i_2 in zip(col2, col3):
                rho+= fin[i_1,j,k] + 2*fin[i_2,j,k]
            rho *= 1/(1-u_0)
        
        # update fluid propagation
        u[0,j,k], u[1,j,k] = u_0, u_1
        
        # iter-wise equilibrium
        for i in range(9):
            cu = 3 * (v[i,0]*u_0 + v[i,1]*u_1)
            usqr = 3/2 * (u_0**2 + u_1**2)
            feq[i] = rho * t[i] * (1 + cu + 0.5*cu**2 - usqr)

        if j == 0 :
            for i in range(3):
                fin[i,j,k] = feq[i] + fin[8-i,j,k] - feq[8-i]
                
        # collision computing
        for i in range(9):
            fout[i,j,k] = fin[i,j,k] - omega * (fin[i,j,k] - feq[i])
            
        # obstacle bounce-back
        if ((j-cx)**2+(k-cy)**2)< (r**2):
            for i in range(9):
                fout[i,j,k] = fin[8-i, j, k]        

@cuda.jit
def stream(fin, fout):
    j, k = cuda.grid(2)
    if j < nx and k < ny:
        for i in range(9):
            fin[i,j,k] = fout[i, (j-v[i,0]) % nx, (k-v[i,1]) % ny]


def main(maxIter, saveat, notebook=False):
    if not notebook:
        print("Initializing Simulation...")
    fin     = cuda.device_array((9,nx,ny))
    fout    = cuda.device_array((9,nx,ny))
    u       = cuda.device_array((2,nx,ny))

    init_simul[blockspergrid, threadsperblock](1, fin)

    if not notebook:
        print("Starting Simulation...")
        figures = {}
        start = tm.time()
    for time in range(maxIter):
        # Cells one-iteration
        cell_iter[blockspergrid, threadsperblock](fin, fout, u)
        stream[blockspergrid, threadsperblock](fin, fout)
        # Recording timestamp velocity
        if (not notebook and time % saveat == 0):
            fluid = u.copy_to_host()
            fig = np.sqrt(fluid[0]**2 + fluid[1]**2).transpose()
            figures[time//saveat] = fig
    
    if not notebook:
        end = tm.time()
        print("Ended in %d seconds." % (end - start))
        print("Saving simulation's visuals...")
        for inst, fig in figures.items():
            imsave("gpu_opt_out/vel.{0:04d}.png".format(inst), fig, cmap="autumn")
        print("DONE! Check ./gpu_opt_out folder for progress visuals.")
    else:
        fluid = u.copy_to_host()
        figure = np.sqrt(fluid[0]**2 + fluid[1]**2).transpose()
        imshow(figure, cmap="autumn")

if __name__ == "__main__":
    main(20000, 1000)
