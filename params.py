import pickle

pkl_filename = "pickle_model.pkl"
print("loading model")
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)
print(" summary ")
print(model.summary())