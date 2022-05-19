# ICL_add_block Package

This package can read or predict (ML) the ICL and BCG classes for a given
stellar population in a main halo of a simulated cluster. After, it can add
this information as a new block to the snapshot file. The writing is general
enough so that these functions can be used to add any block to the snapshot,
not just ICL/BCG labels. 


import ICL_block as b
import RF_ICL_BCG as RF
import numpy as np

#model = RF.load_model(path_model = "Best_Model-Random-3par.pkl")

#dictionary = RF.create_physicals_dictionary(["MCRI","RCRI"], [False, False])

#input_features, phys = RF.data_preparation("/e/ocean2/users/imarini/10x/4Antonio", iFOF = 0, physicals = dictionary)
#classification = model.predict(input_features)

index_ICL_ids, index_BCG_ids = b.get_ids(base_path = "/e/ocean2/users/imarini/10x/4Antonio", read_value=False)

block = b.create_ICL_input_block(index_ICL_ids, index_BCG_ids, base_path ="/e/ocean2/users/imarini/10x/4Antonio", snapnumber = '092', positional = True)

b.add_block("ICL ", block, "/e/ocean2/users/imarini/10x/4Antonio", "/e/ocean2/users/imarini/10x/4Antonio/snapdir_092_prove/", parttype=4)

print ("done")
