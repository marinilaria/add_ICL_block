import sys
sys.path.append("/home/moon/imarini/MODULES")
#from g3read import g3read as g3
#import glob
import numpy as np
import Gadget as g
#import os
import joblib
import pandas as pd
import warnings

def radius(x):
        return np.sqrt(np.sum(x**2,axis=1))
    

def load_model (path_model = 'My_Jar_of_Pickles/Best_Model-Random-3par.pkl'):
    '''
    This function loads the model for the ICL classification. Default is with the best model with
    3 input parameters: (log10(r/r200), v/v200, m200).
    Parameters:
    :: path_model :: (str) path where the model is contained.
    '''
    model = joblib.load(path_model)
    return model

def data_preparation(base_path, iFOF, snapnumber = '092', physicals = None):
    '''
    This function prepares the input features and the ids of the star particles in the simulations (np.unit32), and potentially an array of other interesting quantities as selected by the user, if physicals is True. Note that in this case, the ids are contained in this array.
    Parameters:
    :: base_path ::(str) complete path where simulation snapshot is stored
    :: iFOF :: (int) number of fof to select
    :: snapnumber :: (str, 3 characters) number of snapshot. Default is '092'.
    :: physicals :: (dict) additional properties to get from the simulations. This 
    dictionary can be created by the user or can be obtained by running  
    create_physicals_dictionary(...)
   
    '''
    
    try :
        snapshot = '/snap_' + snapnumber
        snapname = base_path + snapshot
        header = g.snapshot_header(snapname)
        
    except:
        snapshot = '/snapdir_{0}/snap_{0}'.format(snapnumber)
        snapname = base_path + snapshot
        header = g.snapshot_header(snapname)
        
    groupbase = base_path + '/groups_{0}/sub_{0}'.format(snapnumber)

    set_umass = 1e10                 #msun
    set_upos = header.time           #kpc
    set_uvel = np.sqrt(header.time) #km/s
    G =  4.30091e-6                  # kpc Msun^-1 (km / s)^2

    m200 = g.read_block(groupbase, "MCRI")[iFOF] * set_umass
    r200 = g.read_block(groupbase, "RCRI")[iFOF] * set_upos
    gpos = g.read_block(groupbase, "GPOS")[iFOF] * set_upos
    v200 = np.sqrt(G*m200/r200) #km s^-1

    pos = g.read_block(snapname,'POS', parttype = 4) * set_upos - gpos
    m = g.read_block(snapname,'MASS', parttype = 4) * set_umass
    vel = g.read_block(snapname,'VEL', parttype = 4) * set_uvel
    ids = g.read_block(snapname,'ID', parttype = 4)
    
    entro_r200 = np.where(radius(pos)<r200)[0]
    v_cdm = np.average(vel[entro_r200],axis=0,weights=m[entro_r200])
    vel -= v_cdm

    soff = g.read_block(groupbase, "SOFF")
    slen = g.read_block(groupbase, "SLEN")
    pid = g.read_block(groupbase, "PID")
    fsub = g.read_block(groupbase, "FSUB")

    subhalo_ids = pid[soff[fsub[iFOF]]:soff[fsub[iFOF]]+slen[fsub[iFOF]]]
    idd = np.in1d(ids,subhalo_ids,assume_unique=True)
    len_idd = idd.sum() #sum of the True, gives the number of stars in main halo
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        input_features = pd.DataFrame({'distance':np.log10(radius(pos[idd])/r200),'velocity':radius(vel[idd])/v200,'logm200':[np.log10(m200*1e-14)]*len_idd})
        input_features = input_features.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    
    if physicals is not None:
        physicals["ID  "] = True
    else:
        physicals = {"ID  ": True}

    array_physicals = []
    for physical, is_snapshot in physicals.items():
        if is_snapshot:
                dummy = g.read_block(snapname, physical,parttype = 4)
                array_physicals.append(dummy[idd]) 
        else:
                dummy = g.read_block(groupbase, physical)
                array_physicals.append([dummy[iFOF]]*len_idd)
            
    array_physicals = pd.DataFrame(np.reshape(array_physicals, (len(physicals.keys()), len_idd)).T, columns = physicals.keys())
    array_physicals = array_physicals.iloc[input_features.index].reset_index(drop = True)
    input_features = input_features.reset_index(drop = True)
        
    return input_features, array_physicals
        
def create_physicals_dictionary(keys, is_snapshot):
    '''
    This function creates a dictionary necessary if one wants to get a series of properties from the
    simulation other than the input parameters in data_preparation(...).
    Parameters:
    :: keys :: (array of str) names of the properties as reported in the Gadget INFO block
    :: is_snapshot :: (array of bool) If True the property is in the snapshot, if False is Subfind.
    '''

    assert len(keys) == len(is_snapshot), "Number of keys needs to be equal to number in is_snapshot"
    dictionary = {}

    for key,is_snap in zip(keys,is_snapshot):
        dictionary[key] = is_snap
    
    return dictionary

def get_indices(ids, class_list):
        '''
        This function returns an array containing the positional indices of class_list in ids. This means that rather than having ids, we have the indices of where they are in ids.  
        Parameters:
        :: ids :: (array) contains the ids of the particles
        :: class_ids :: (array) contains the ids of the selected particles
        ''' 
        new_ids = np.arange(len(ids))
        class_ids = np.in1d(ids, class_list, assume_unique=True)
        return new_ids[class_ids]








