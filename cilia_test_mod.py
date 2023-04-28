import numpy as np
from fluid_functions_copy import friction
from fluid_functions_copy import change_basis
from fluid_functions_copy import Blake_Tensor

from itertools import product
import matplotlib.pyplot as plt

from scipy import optimize
from scipy.integrate import solve_ivp

from joblib import Parallel, delayed
# To get parallel for loops


f = lambda phi : 1 

#Sample background flow
bf= lambda pos: np.array([0, 0 , 0])

#np.array([0.5* pos[-1]**2, 0 , 0])


def big_phi(phi, lamb):
    
    '''Maps phi (array) -> Big phi (array)'''
    '''This is a circle homeomorphism har har'''
    
    '''This relies on the data from the limit cycle found in 3e'''
    
    arr= np.genfromtxt(f'limit_cycles/{lamb}_lamb.csv', delimiter=',')
    
    p= arr[1][1:]
    r= arr[2][1:]
    
    ''' Hacky function which finds to find index of point that starts at pi/2 '''
    def est_time(L1):
        L= np.array(L1) % (2*np.pi)
        low= next(x[0] for x in enumerate(L) if x[1] < 0.001)
        high= next(x[0] for x in enumerate(L[low:]) if x[1] > 6.28305)
        return high+2, low

    def limit_arr2(p_list, r_list):
        '''Extracts limit cycle given list'''
        a,b= est_time(p_list)
        return np.linspace(0,1, a), p_list[b:b+a]
    
    x, y= limit_arr2(p, r)
    
    l_space= np.linspace(0, 2*np.pi, len(y))
    
    adj= (y- l_space -y[0])
    p= np.array(phi)
    
    '''Returns big phi based on similar phi in list'''
    
    index= (p%(2*np.pi)/ (2*np.pi) * len(y)).astype(int)
    
    return phi - adj[index]  

class Cilium:
    
    def __init__(self, lattice_position , coordinates, constants= np.array([0.5,1,1]) , force= f, h=1, a=0.01):
        
        '''Define characteristic of each cilia'''
        
        #Lattice position is in x,y plane
        self.lp= lattice_position
        
        #Coordinates is in zeta, r, phi
        self.coor= coordinates
        
        #Constants should be r0, l_ambda, eta
        self.constants= constants
        
        self.height= h
        self.force = force
        
        self.a_rad= a
            
    @property
    def pos(self):
        
        ''' 
        Input: Cilia
        Output: position in x, y, z
        '''
        
        zeta, r , phi= self.coor
        
        h= self.height
        
        return np.append(self.lp, [h]) + np.array([r*np.sin(phi), zeta, r*np.cos(phi)])
    
    @property
    def basis_change(self):    
        
        '''Change of basis applied to each cilium'''
        
        return change_basis(self.coor[-1])
        
    
    @property
    def f_vector(self):
        
        '''Calculates the force vector wrt x y z coor'''
        
        zeta, r, phi = self.coor
        r0, lamb, eta = self.constants
        
        force= self.force
        
        F= np.array([force(phi) , -lamb*(r-r0), -eta*zeta])
        
        return F
    
    @property
    def friction(self):
        
        '''Calculates friction given position of cilium'''
        
        return friction(self.pos, a= self.a_rad)
    
class Cilia_System:
    
    def __init__(self, background_flow=bf):
        
        self.list= []
        self.len= len(self.list)
        self.background= background_flow
        
        pass
    
    def add(self, Cilias):
        
        '''Add cilias to couple into the system'''
        
        for c in Cilias:
            self.list.append(c)
            
    def d_coor(self):
        
        '''finds velocity given the current state'''
        
        force_vectors= [cil.f_vector for cil in self.list]
        basis_changes= [cil.basis_change for cil in self.list]
        pos_list= [cil.pos for cil in self.list]
        
        n= self.len
        
        vel_vec=[]
        
        def d_coor_cil(index):
            
            pos_i= pos_list[index]
            cil_i= self.list[index]
            zeta, r , phi= cil_i.coor
            
            v_cil=  np.linalg.inv(friction(pos_i)) @ basis_changes[index] @ force_vectors[index]

            for j, cil_j in enumerate(self.list):
                if index!=j:
                    pos_j= pos_list[j]
                    v_cil+= Blake_Tensor(pos_i, pos_j) @ basis_changes[j] @force_vectors[j]
            
            dr_phi, dr, dz= cil_i.basis_change.T @ v_cil
            d_coor= np.array([dz, dr , dr_phi /r])
            
            return d_coor
        
        return [d_coor_cil(i) for i in range(len(self.list))]
    
        #return Parallel( n_jobs = 4)(delayed(d_coor_cil)(i) for i in range(len(self.list)))
        '''^Parallel implementation'''
        
    

    def update(self, stepsize=0.01):
        
        '''Simulates 1 timestep (scales with O(n^2) with number of cilia) '''
        
        force_vectors= [cil.f_vector for cil in self.list]
        basis_changes= [cil.basis_change for cil in self.list]
        pos_list= [cil.pos for cil in self.list]
        
        #background_flow= self.background

        n= self.len
        
        vel_vec=[]
        
        #Double for loops 
        
        for i, cil_i in enumerate(self.list):
            
            # Calculate for stokelet i 
            
            pos_i= pos_list[i]
            zeta, r , phi= cil_i.coor
            
            # Base velocity in absence of other Stokelets
            v_cil=  np.linalg.inv(friction(pos_i)) @ basis_changes[i] @ force_vectors[i]
            
            interactions= np.zeros(3) 
            
            # Incooperate hydrodynamic interactions of other Stokelets
            for j, cil_j in enumerate(self.list):
                
                if i!=j:
                    pos_j= pos_list[j]
                    interactions+= Blake_Tensor(pos_i, pos_j) @ basis_changes[j] @force_vectors[j]
                    
            #+ background_flow(cil_i.pos)
                                            
            dr_phi, dr, dz= cil_i.basis_change.T @ (v_cil + interactions)
            d_coor= np.array([dz, dr , dr_phi /r])
            
            cil_i.coor += d_coor * stepsize
                        
    @property
    def phi_list(self):
        return np.array([cil.coor[-1] for cil in self.list])
    
    @property
    def r_list(self):
        return np.array([cil.coor[1] for cil in self.list])
    
    @property
    def zeta_list(self):
        return np.array([cil.coor[0] for cil in self.list])
    
    @property
    def coor_list(self):
        
        zeta_l= np.array([cil.coor[0] for cil in self.list])
        r_l= np.array([cil.coor[1] for cil in self.list])
        phi_l= np.array([cil.coor[2] for cil in self.list])
        
        return zeta_l, r_l, phi_l
    
    def flow(self, pos):
        
        '''Calculates the velocity vector'''
            
        return np.sum([Blake_Tensor(pos, cil_j.pos) @ cil_j.f_vector for cil_j in (self.list)], axis=0)
    
    vflow = np.vectorize(flow)
        
    def plot_flow(self, y_slice = 0, x_range= (0,1), z_range = (0,1) , grid=20):
        
        '''Generates plot of flow velocities at a slice (y=0)'''
        
        x_min, x_max = x_range
        z_min, z_max = z_range
        
        x = np.linspace(x_min, x_max, grid)
        z = np.linspace(z_min, z_max, grid)
        
        X,Z = np.meshgrid(x, z)
        
        xv_list=[]
        zv_list=[]
        
        background_flow= self.background
        
        for xs, zs in product(x,z):
            
            flow_v= self.flow(np.array([xs, y_slice , zs])) + background_flow(np.array([xs, y_slice, zs]))
            
            xv_list.append(flow_v[0])
            zv_list.append(flow_v[2])
        
        p_x = []
        p_z = []
        
        for cil_i in self.list:
            
            position = cil_i.pos
            p_x.append(position[0])
            p_z.append(position[2])
        
        
        fig, ax = plt.subplots(figsize =(14, 8))
            
        ax.quiver(Z, X, xv_list, zv_list)
        ax.scatter(p_x, p_z, c='red', s=200)
        
        plt.grid()
        # show plot
        plt.show()

    def plot_2D_phi(self, index, dim= [16,8]):
        
        '''Plot 3D surface of sin phi used to generate gifs'''
        
        x_n= dim[0]
        y_n= dim[1]
        
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize = (10, 10))
        
        X= np.reshape([cil.lp[0] for cil in self.list], (x_n, y_n))
        Y= np.reshape([cil.lp[1] for cil in self.list], (x_n, y_n))
        
        Z= np.sin(np.reshape([cil.coor[-1] for cil in self.list] , (x_n, y_n)))

        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
        
        ax.set_zlim(-1.5, 1.5)
        
        ax.set_zlabel("Sin(phi)")
        ax.set_xlabel("x_coor")
        ax.set_ylabel("y_coor")
        
        ax.set_title(f"Phases at timestep={index}")
        plt.grid()

def gen_system(X, lattice, forcings, const= [0.5, 0.1 ,10] ):
    '''Generate a cilia system given a pre-defined state '''
    
    ''' ie. X[0] -> state of cilia 1 '''
    n= len(X)
    
    cil_list= [Cilium(lattice[i], X[i], constants= const, force= forcings[i]) for i in range(n) ]
    
    Cilia_System_1= Cilia_System()
    Cilia_System_1.add(cil_list)
        
    return Cilia_System_1

def d_core(t,X, lattice, forcings, const= [0.5, 0.1 ,10] ):

    #Generates velocities given a state (X) in an array form.
    #t is a dummy variable added to work nicely with the scipy's solve_ivp function.

    n=len(lattice)
    
    X=np.array(X).reshape((n,3))
    c_sys= gen_system(X, lattice, forcings, const )
    
    return np.array(c_sys.d_coor()).flatten()

def flow_map(tf, X0, lattice, forcings, cons= [0.5, 0.1 ,10], maxstep=0.1):

    '''Integrates the system forwards in time given initial state X0'''
    
    sol= solve_ivp(lambda t, y: d_core(t, y, lattice, forcings, cons), t_span=(0, tf), y0=X0, max_step= maxstep)
    
    solve= sol.y.T[-1]

    #Mod 2-pi the angles.
    
    for i in range(len(lattice)):
        solve[i*3 + 2] = solve[i*3 + 2] % (2*np.pi)
                   
    return solve

def b_residual(t,X, lattice, forcings, const= [0.5, 0.1 ,10] ):
    '''residual 'b' in Ax=b for the JFNK method'''
    
    return np.dot(d_core(t, X, lattice, forcings, const) , X)

def aug_residual(X_aug, lattice, forcings, cons= [0.5, 0.1 ,10], max_step=0.1):
    
    '''Augmented residual to find periodic orbit'''
    
    t, X1 = X_aug[0] , X_aug[1:]
    n= int(len(X1)/3)
    
    res= np.append([t], flow_map(t, X1, lattice, forcings, cons, maxstep=max_step))- np.array(X_aug)
    
    res[0]= b_residual(t,X1, lattice, forcings, cons)
    
    for i in range(n):
        res[3*i+3]= min(np.abs(res[3*i+3] % (2*np.pi)), np.abs(2*np.pi - res[3*i+3] % (2*np.pi)))   
        
    return res