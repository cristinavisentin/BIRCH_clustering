from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from CFTree import CFTree
from CF import ClusteringFeature
import csv
import sys
from sklearn.linear_model import LinearRegression


class BIRCH:
    def __init__(self, data_dim=2, page_size=256, threshold=0.0, max_size=100_000) -> None:
        self.tree = CFTree(data_dim, page_size, threshold, True)
        self.max_size = max_size
        self.data_dim = data_dim
        self.page_size = page_size
        self.threshold = threshold
        self.max_size = max_size
        self.old_thresholds: list[list[float]] = [[.0001]]
        self.N_added_list: list[list[int]] = []
        self.radii: list[list[float]] = []
        self.N = 100_0000

    def addall(self, nodefile):
        with open(nodefile, "rt") as f:
            reader = csv.reader(f)
            for row, text in enumerate(reader):
                # controllo ogni 1000 righe che l'albero non e' troppo grande in MB
                if sys.getsizeof(self.tree) > self.max_size:
                    # ricalcola threshold
                    threshold = self.recompute_threshold(row)
                    # comprimi
                    self.compress(threshold)
                # aggiungo un nodo
                coordinates = np.array(text, dtype=float)
                cf = ClusteringFeature(self.data_dim).add_point(coordinates)
                self.tree.insert_CF(cf)
            return self

    def recompute_threshold(self, row: int):
        # stima quanti dati vogliamo aggiungere alla prossima iterazione
        N_next = min(self.N, 2 * row)
        # calcola il raggio della root
        root = self.tree.compute_cumulative_CF()
        radius = root.radius()
        self.radii.append([radius])
        self.N_added_list.append([row])

        model = LinearRegression()
        # packed volume increases linearly with threshold
        # packed volume Vp = T ** d
        N_power = [[number[0] ** (1/self.data_dim)]
                   for number in self.N_added_list]
        # stima la prossima threshold
        model.fit(N_power, self.old_thresholds)
        next_threshold = model.predict(np.array(N_next).reshape(-1, 1))[0][0]
        # stima il prossimo raggio
        model.fit(self.N_added_list, self.radii)
        radius = model.predict(np.array(N_next).reshape(-1, 1))[0][0]

        expansion_factor = max(1.0, float(radius / self.radii[-1][0]))
        # trova nel cluster con più nodi i leaf node più vicini
        d_min = self.tree.find_smallest_increase()

        next_threshold = min(d_min, float(expansion_factor * next_threshold))
        # se non è abbastanza grande incrementala
        if next_threshold <= self.old_thresholds[-1][0]:
            next_threshold = next_threshold*(N_next/row)**(1/self.data_dim)
        self.old_thresholds.append([next_threshold])
        return next_threshold

    def compress(self, threshold):
        # if threshold < self.threshold:
        #     command = input(
        #         "Sicuro che vuoi procedere con una threshold inferiore?").lower().strip
        #     if command == "yes":
        #         self.threshold = threshold
        #     else:
        #         return self
        # else:
        self.threshold = threshold
        compressed = CFTree(self.data_dim, self.page_size,
                            self.threshold, True)
        for path in self.tree.paths():
            # compressed.add_path_from_other(path, self.tree)
            for leaf in self.tree[path].leaves():
                compressed.insert_CF(leaf)
            self.tree.remove(path)
        self.tree = compressed
        return self


# # Generiamo dataset
# X, y = make_blobs(
#     n_samples=100000, centers=10, random_state=42)
# # Aggiungiamo un po' di noise
# noise = np.random.uniform(low=-10, high=10, size=(5000, 2))
# X = np.vstack([X, noise])
# with open("data.csv", "tw") as f:
#     writer = csv.writer(f)
#     writer.writerows(X)

model = BIRCH()
model.addall("data.csv")
print(model.tree)
