import h5py

def save_hdf5_matrix(filename,x):    
    print(filename)
    file = h5py.File(filename,'w')
    file.create_dataset("Default", data=x)        
    file.close()
        
def load_hdf5_matrix(filename):    
    f = h5py.File(filename,'r')
    
    z = f['Default'][:]
    
    f.close()
    return z
