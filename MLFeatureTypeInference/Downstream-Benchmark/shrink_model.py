import pickle
import _pickle as cPickle
import dill
import bz2

print("loading original pickle file...")
# Load original pickled model
rf_filename = "resources/RandForest.pkl"
with open(rf_filename, 'rb') as f:
    model = pickle.load(f)

# print("saving dill pickle file...")
# # Save the model with dill
# dill_filename = "resources/DillRF.pkl"
# with open(dill_filename, "wb") as f:
#     dill.dump(model, f)

# print("loading dill pickle file...")
# # Load the dill pickled model
# with open(dill_filename, 'rb') as f:
#     dill_model = dill.load(f)

def compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f: 
        cPickle.dump(data, f)

def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data

compressed_pickle("resources/comp_rf", model)
new_model = decompress_pickle("resources/comp_rf.pbz2")

# print(type(model))
# print(type(new_model))