from sklearn.neighbors import NearestNeighbors
import numpy as np
import torch


promoted = np.array([10,3,7])
rating = torch.randn((5, 3))
value, indice = torch.topk(rating, k=1)
indice = indice.cpu().numpy()
topk = promoted[indice]
print("1")
