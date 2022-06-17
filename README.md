# ICL_add_block Package

This package can read or predict (ML) the ICL and BCG classes for a given
stellar population in a main halo of a simulated cluster. After, it can add
this information as a new block to the snapshot file. The writing is general
enough so that these functions can be used to add any block to the snapshot,
not just ICL/BCG labels. 

## Example code
```
import ICL_block as b
import RF_ICL_BCG as RF
import numpy as np

#model = RF.load_model(path_model = "Best_Model-Random-3par.pkl")

#dictionary = RF.create_physicals_dictionary(["MCRI","RCRI"], [False, False])

#input_features, phys = RF.data_preparation("dir/", iFOF = 0, physicals = dictionary)
#classification = model.predict(input_features)

index_ICL_ids, index_BCG_ids = b.get_ids(base_path = "dir/", read_value=False)

block = b.create_ICL_input_block(index_ICL_ids, index_BCG_ids, base_path ="dir/", snapnumber = '092', positional = True)

b.add_block("ICL ", block, "dir/", "snapdir_new/", parttype=4)

print ("done")
```
