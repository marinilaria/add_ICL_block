from g3read import g3read as g3
import glob
import numpy as np
import Gadget as g
import os
import shutil
import RandomForest as RF
import pandas as pd

def get_ids(base_path, read_value = True,  iFOF = 0, snapnumber = 92,  physicals = None ):
    '''
    This function returns the ICL (or BCG) ids in a given simulation. 
    Parameters:
    :: read_value :: Switch to control whether the particles classifications are read from a file 
    (True) or deducted from the random forest classifier (False). Default is True.
    :: base_path :: (str) path where the files are contained. If read_value is True, the file refers
    to the ICL ids: first one is the ICL, second is the BCG. If read_value is False, read the path 
    where simulations are contained (before snapdir/groups level). Default is None.
    :: iFOF :: (int) number of fof to select
    :: snapnumber :: (integer) number of snapshot. Default is 92.
    :: physicals :: (dict) additional properties to get from the simulations. This
    dictionary can be created by the user or can be obtained by running
    create_physicals_dictionary(...)
    '''
    if (read_value == True) :
        #read the files
        assert len(base_path) == 2, "Provide path file for ICL and BCG."
        index_ICL_ids = np.loadtxt(base_path[0])
        index_BCG_ids = np.loadtxt(base_path[1])
    else:
        #calculate the model
        input_features, physicals = RF.data_preparation(base_path, iFOF, snapnumber = snapnumber,  physicals = physicals) 
        ids_input_features = physicals["ID  "]
        model = RF.load_model("Best_Model-Random-3par.pkl")
        classification = model.predict(input_features.values)
        index_ICL_ids = ids_input_features.iloc[classification == 0]
        index_BCG_ids = ids_input_features.iloc[classification == 1]
        
    return index_ICL_ids, index_BCG_ids

def create_ICL_input_block(index_ICL_ids, index_BCG_ids, base_path, snapnumber = 92, positional = True, parttype = 4):

    ''' 
    This function creates an ICL_array containing the classification of particles in the simulation: 
    whether they are ICL (0), BCG (1) or neither (-1). Notice that to get the particles in the
    satellites one should just select the stars in this array, and therefore -1 would be corresponding
    to the stars in the satellite.
    Parameters:
    :: ICL_ids :: (array: uint32/int32) ids of the particles in the ICL
    :: BCG_ids :: (array: uint32/int32) ids of the particles in the BCG
    :: base_path :: (str) where simulations are contained, up to the snapdir level
    :: snapnumber :: (integer) snapshot number 
    :: positional :: if False, ICL_ids and BCG_ids are providing the indices of the particles
    as ordered in the simulations. If True, ICL_ids and BCG_ids are providing the IDS of this particles
    that still need to be matched to the IDS in the simulation. Default is True
    '''
    #consistency with Gadget read Giuseppe Murante

    try :
        snapshot = '/snap_' + snapnumber
        snapname = base_path + snapshot
        ids = g.read_block(snapname, "ID", parttype = parttype)

    except:
        snapshot = '/snapdir_{0:03}/snap_{0:03}'.format(snapnumber)
        snapname = base_path + snapshot
        ids = g.read_block(snapname, "ID", parttype = parttype)                            
    
    
    
    if positional:
        ICL_ids = np.uint32(index_ICL_ids)
        BCG_ids = np.uint32(index_BCG_ids)
        mask_ICL = np.in1d(ids, ICL_ids)
        mask_BCG = np.in1d(ids, BCG_ids)
    else:
        mask_ICL = index_ICL_ids
        mask_BCG = index_BCG_ids
        
    len_ids = len(ids)
    class_array = -np.ones(len_ids, dtype = np.float32) #all particles
    class_array[mask_BCG] = 1 ##particles BCG == 1
    class_array[mask_ICL] = 0 ##particles ICL == 0
    
    return class_array


def add_block(blockname, data, base_path, base_save_path, parttype = 4, snapnumber = 92, dim = 1):
    '''
    This function adds a block to an exisisting snapshot file (format 2) from Gadget.
    At the moment it works only for parttype = -1, due to some conflicts in g3read routine. Check for
    future updates.

    Parameters:
    :: blockname :: (str: 4characters) the name you want to assign to your block (e.g. : "ICL ")
    :: data ::      (array) the data you want to assign to the column. Length of the data
    needs to be consisted with the particle type. This means that if I have N stars, I need
    to assign N values to that particle type.
    :: base_snap_path :: (str) where simulations are contained, up to the snapdir level
    :: base_save_path :: (str) where new snapshots will be saved, up to the snapdir level
    :: parttype ::  (int) to which type of particle assign the data, options: -1 (all)
    or either one from [0,5]
    :: snapnumber :: (int) the snapshot number
    :: dim :: (int) if the dimension of the data is different than 1 (e.g. pos would have dim = 3)
    '''
    try :
        snapshot = '/snap_' + snapnumber
        snapname = base_path + snapshot
        snapshot_files = np.array(glob.glob(snapname + '*'))
        
        #save only snapshot files
        mask = np.array([string[-1].isdigit() for string in snapshot_files])
        snapshot_files = np.sort(snapshot_files[mask])
        
    except:
        snapshot = '/snapdir_{0:03}/snap_{0:03}'.format(snapnumber)
        snapname = base_path + snapshot
        snapshot_files = np.array(glob.glob(snapname + '*'))
        
        #save only snapshot files
        mask = np.array([string[-1].isdigit() for string in snapshot_files])
        snapshot_files = np.sort(snapshot_files[mask])

    #Check size of data
    len_data = len(g.read_block(snapname ,"MASS", parttype = parttype))
    if (len(data)!=len_data):
        raise SystemExit("The input data is not of the same length as the data in the snapshot for the given particle type(s) ")
        

    start = 0
    
    for snapshot_file in snapshot_files:
        header = g3.GadgetFile(snapshot_file).header
        if (parttype > -1):
            npart = int(header.npart[parttype])
        else:
            npart = int(header.npart.sum())
            
        filename = base_save_path + os.path.basename(os.path.normpath(snapshot_file))
    
        shutil.copyfile(snapshot_file, filename)
        f = g3.GadgetFile(filename)
        f.add_file_block(blockname, data.dtype.itemsize * npart * dim, partlen = data.dtype.itemsize * dim, ptypes = parttype)
        
        f.write_block(blockname, parttype, data[start:start + npart])        
        start += npart
    return
