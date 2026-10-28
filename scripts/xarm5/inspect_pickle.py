import os
import pickle
import numpy as np

if __name__ == "__main__":
    path = os.path.expanduser("~/data/bc_data/gello/000/00001.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    for key, value in data.items():
        print(f"{key}: {type(value).__name__}")
        
        if isinstance(value, np.ndarray):
            print(f"  Shape:      {value.shape}")
            print(f"  Dtype:      {value.dtype}")
            print(f"  RAM:        {value.nbytes / 1024 / 1024:.4f} MB")
            
        serialized = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Pickled:    {len(serialized) / 1024 / 1024:.4f} MB\n")
    # print(data.keys())