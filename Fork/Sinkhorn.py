# Reload later
import pickle
import matplotlib.pyplot as plt
with open('data2.pkl', 'rb') as f:
    fig = pickle.load(f)
plt.show()