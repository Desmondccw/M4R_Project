import numpy as np

def Blake_Tensor(R_1, R_2, eta=100):
    
        '''
        Input: R_1, R_2 (Stokeslet positions)
        Output: Blake Tensor (represents hydrodynamic coupling)
        '''
        
        '''Define d'''
    
        d= R_1 - R_2
        d_norm= np.linalg.norm(d)
        
        #image tensor
        x1, y1, z1= R_1
        x2, y2, z2= R_2
    
        '''Define the image'''
    
        d_img= R_1 - np.array([x2, y2, -z2])
        d_img_norm= np.linalg.norm(d_img)
    
        d3= d_img_norm**3
        d5= d_img_norm**5
        
        '''Contribution from reflections eg..(Stresslet + Sourcelet)'''
        
        im= np.zeros((3, 3))
    
        im[0][0], im[1][1]= -2*z1*z2 * (1/d3 - 3* (np.array([x1-x2, y1-y2]))**2 /d5)
        im[2][2]= 2*z1*z2 * (1/d3 - 3*(z1+ z2)**2 /d5)
        
        im[1][0]= im[0][1] = 6 *z1 *z2 *(x1- x2) * (y1 -y2)/d5
        im[0][2] , im[2][0] = 2*(x1- x2)* (z1**2/d3 + 3 * z1 *z2 * (z1 + z2)* np.array([-1, 1]) / d5) 
        im[1][2] , im[2][1] = 2*(y1- y2)* (z1**2/d3 + 3 * z1 *z2 * (z1 + z2)* np.array([-1, 1]) / d5)
        
        '''Combine Greens function'''
        G= im + (1/d_norm - 1/d_img_norm) * np.identity(3) + np.outer(d,d)/d_norm**3 - np.outer(d_img,d_img)/d3
                                
        return G /(8*np.pi*eta)
    
def friction(position, vis= 100, a= 0.01): 
    '''
    Input: x, y , z coord(Stokeslet position)
    Output: Friction tensor (up to O((a/z)^3))
    '''
    
    x, y , z = position
    
    base= 6 * np.pi * vis * a
    
    I= np.identity(3)
    e_z=np.array([0,0,1])

    return base * ( I + (9*a /(16*z)) * ( I + np.outer(e_z, e_z)))

def change_basis(phi):
    
    "Change of basis matrix"
    
    '''x, y, z -> r , phi,zeta '''
    
    '''Input: list of phases of cilia'''
    '''Output: Change of basis matrix '''
    
    p= phi
    
    R= np.array([[np.cos(p), np.sin(p), 0], 
                        [0          , 0          , 1], 
                        [-np.sin(p), np.cos(p), 0]])
    
    return R