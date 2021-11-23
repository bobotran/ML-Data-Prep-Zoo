import pickle

print("loading original pickle file...")

# Load original pickled model
rf_filename = "resources/RandForest.pkl"
with open(rf_filename, 'rb') as f:
    model = pickle.load(f)

def split_model(source_model_file, write_size):
    cur_num = 0
    input_file = open(source_model_file, 'rb')
    while True:
        chunk = input_file.read(write_size)
        if not chunk:
            break
        cur_num += 1
        filename = f'resources/split_model_{cur_num}'
        dest_file = open(filename, 'wb')
        dest_file.write(chunk)
        dest_file.close()
    input_file.close()
    return partnum

# Rejoins the model file locally
def rejoin_model(dest_file, read_size):
    output_file = open(dest_file, 'wb')
    parts = ['resources/split_model_1', 'resources/split_model_2']
    for file in parts:
        path = file
        input_file = open(path, 'rb')
        while True:
            bytes = input_file.read(read_size)
            if not bytes:
                break
            output_file.write(bytes)
        input_file.close()
    output_file.close()

# Ex Usage:
# Split file into 2 parts
split_model(source="resources/RandForest.pkl", write_size=60000000) # ~60 mb
# Rejoin the model locally
rejoin_model(dest_file="resources/full_rf.pkl", read_size=60000000) # ~60 mb

# Test loading the rejoined split model
full_rf = "resources/full_rf.pkl"
with open(full_rf, 'rb') as f:
    new_model = pickle.load(f)

print(type(model))
print(type(new_model))

print(model)
print(new_model)