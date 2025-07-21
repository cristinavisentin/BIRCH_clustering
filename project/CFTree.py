import sys
from CF import ClusteringFeature
import copy
import numpy as np


class CFTree:
    def __init__(self, data_dimensionality, branching_factor, threshold, is_leaf: bool, parent=None):
        self.branching_factor = branching_factor
        self.threshold = threshold
        self.is_leaf = is_leaf
        self.data_dimensionality = data_dimensionality
        self.parent = parent
        self.CF_list: list[ClusteringFeature] = []
        self.children_list: list['CFTree'] = []

    def insert_CF(self, cf: ClusteringFeature):
        # 'closest_cf' è un CF, il più vicino al CF passato in input
        closest_cf = self.find_closest_CF(cf)
        # Se nodo leaf
        if self.is_leaf:
            # Se non ci sono ancora CF nel nodo leaf
            if not self.CF_list:
                self.CF_list.append(cf)
                return

            # Provo a vedere se il più vicino e quello in input < T
            if closest_cf:
                temp_cf = copy.copy(closest_cf)
                temp_cf.merge(cf)

                if temp_cf.radius() < self.threshold:
                    # Se si, we are good to go, aggiungo e basta
                    closest_cf.merge(cf)
                else:
                    # Altrimenti aggiungo il CF al nodo
                    self.CF_list.append(cf)
        # Se il nodo non è foglia
        else:
            if closest_cf:
                # Mi trovo l'indice del CF più vicino a quello dato in input
                idx = self.CF_list.index(closest_cf)
                # Attraverso l'indice trovo il nodo figlio (che è un CFNode)
                child = self.children_list[idx]

                # Inserisco il CF nel figlio, aka chiamo ricorsivamente questo metodo
                child.insert_CF(cf)

                if len(child.CF_list) > self.branching_factor:
                    seed1, seed2 = self.split_child(idx)
                    if self.parent and len(self.parent.children_list) <= self.branching_factor:
                        self.merge_refinement(seed1, seed2)
                else:
                    self.CF_list[idx] = self.children_list[idx].compute_cumulative_CF()
        # controlla che la root non e' troppo grande
        if not self.parent and len(self.CF_list) > self.branching_factor:
            # creane una nuova
            new_root = CFTree(
                self.data_dimensionality, self.branching_factor, self.threshold, False)
            self.parent = new_root
            new_root.CF_list.append(self.compute_cumulative_CF())
            new_root.children_list.append(self)
            # dividi quella vecchia
            new_root.split_child(0)
            self = new_root

    def _find_index(self, mode="furthest"):
        op = 1.0 if mode == "furthest" else -1.0
        seed1 = 0
        seed2 = 0
        max_dist = -float("inf")
        for i in range(len(self.CF_list)-1):
            for j in range(i + 1, len(self.CF_list)):
                dist = op * self.CF_list[i].dist(self.CF_list[j])
                if dist > max_dist:
                    max_dist = dist
                    # seed1 e seed2 sono due CF
                    seed1 = i
                    seed2 = j
        return seed1, seed2

    def split_child(self, idx: int) -> tuple[int, int]:
        child = self.children_list[idx]
        # Find i nodi più lontani
        seed1, seed2 = child._find_index(mode="furthest")

        # Step 2: crea due nuovi nodi
        params_dict = {"data_dimensionality": self.data_dimensionality,
                       "branching_factor": self.branching_factor,
                       "threshold": self.threshold,
                       "is_leaf": child.is_leaf,
                       "parent": self
                       }
        new_node1 = CFTree(**params_dict)
        new_node2 = CFTree(**params_dict)
        # inserisco i seed ciascuno in un nuovo nodo
        new_node1.CF_list.append(child.CF_list[seed1])
        new_node2.CF_list.append(child.CF_list[seed2])
        if not child.is_leaf:
            # hanno figli
            new_node1.children_list.append(child.children_list[seed1])
            new_node2.children_list.append(child.children_list[seed2])

        # inserisco i restanti
        for node_idx, cf in enumerate(child.CF_list):
            if node_idx not in (seed2, seed1):
                dist1 = cf.dist(new_node1.CF_list[0])
                dist2 = cf.dist(new_node2.CF_list[0])
                if dist1 < dist2:
                    # inserisco nel nuovo tree
                    new_node1.CF_list.append(cf)
                    if not child.is_leaf:
                        # allora ha figli e li inserisco
                        new_node1.children_list.append(
                            child.children_list[node_idx])
                else:
                    new_node2.CF_list.append(cf)
                    if not child.is_leaf:
                        new_node2.children_list.append(
                            child.children_list[node_idx])
        # aggiungo i nuovi nodi ai children di questo
        self.children_list.append(new_node1)
        self.CF_list.append(new_node1.compute_cumulative_CF())
        self.children_list.append(new_node2)
        self.CF_list.append(new_node2.compute_cumulative_CF())
        # rimuovo la reference di quello vecchio
        self.CF_list.pop(idx)
        self.children_list.pop(idx)

        return seed1, seed2

    # maybe aggiungi un modo per calcolare la distanza in modo custom
    def find_closest_CF(self, cf: ClusteringFeature) -> ClusteringFeature | None:
        min_dist = float('inf')
        closest = None
        for entry in self.CF_list:
            dist = np.linalg.norm(entry.centroid() - cf.centroid())
            if dist < min_dist:
                min_dist = dist
                closest = entry
        return closest

    def insert_datapoint(self, point: np.ndarray):
        pass
        # Creo il CF
        cf = ClusteringFeature(self.data_dimensionality)
        cf.add_point(point)

        # Lo inserisco nell'albero
        split_result = self.insert_CF(cf)

        if split_result:
            self.root = split_result

    def compute_cumulative_CF(self):
        cf = ClusteringFeature(self.data_dimensionality)
        for entry in self.CF_list:
            cf.merge(entry)
        return cf

    def remove(self, path: list[int]):
        """Remove the node at the end of the path"""
        current_node = self
        for node in path:
            if current_node.children_list:
                current_node = current_node.children_list[node]
            else:
                current_node = None
                return self
        return self

    def add_path_from_other(self, path: list[int], other: "CFTree"):
        """Add the nodes on path from other onto this CFTree"""
        current_node = other
        for node in path:
            self.insert_CF(current_node.CF_list[node])
            current_node = copy.copy(current_node.children_list[node])
        return self

    def find_smallest_increase(self) -> float:
        if self.is_leaf:
            # find closest nodes
            seed1, seed2 = self._find_index("smallest")
            dist = self.CF_list[seed1].dist(self.CF_list[seed2])
            # return their distance
            return dist
        # else find most dense child
        idx_max = np.argmax(list(map(lambda x: x.N, self.CF_list)))
        return self.children_list[idx_max].find_smallest_increase()

    def merge_refinement(self, split1, split2):
        # Step 1: trova i due CF più vicini
        seed1, seed2 = self._find_index("closest")
        # se non sono sia split1 che split2
        if split1 != seed1 or split2 != seed2:
            temp_cf = copy.copy(self.CF_list[seed1])
            temp_cf.merge(self.CF_list[seed2])
            # se posso unirli
            if temp_cf.radius() < self.threshold:
                # inserisci le foglie del secondo nel primo
                for leaf in self.children_list[seed2].leaves():
                    self.children_list[seed1].insert_CF(leaf)
                # svuota spazio
                self.children_list.pop(seed2)
                self.CF_list.pop(seed2)
                # se troppo grandi splitta
                if len(self.children_list[seed1].children_list) > self.branching_factor:
                    # serve un nuovo split
                    self.split_child(seed1)
        return self

    def paths(self, path: list[int] = []):
        """Iterable returning list of all possible paths in the tree"""
        if self.is_leaf:
            yield path
        else:
            for idx, node in enumerate(self.children_list):
                copied = path + [idx]
                yield from node.paths(copied)

    def leaves(self):
        if self.is_leaf:
            for cf in self.CF_list:
                yield cf
        else:
            for child in self.children_list:
                yield from child.leaves()

    def __getitem__(self, path):
        current_node = self
        for node in path:
            current_node = current_node.children_list[node]
        return current_node

    def __repr__(self, level=0) -> str:
        to_return = ""
        indent = "----" * level
        for idx, node in enumerate(self.CF_list):
            to_return += f"{indent}- {node}\n"
            if self.children_list:
                child = self.children_list[idx]
                to_return += child.__repr__(level + 1)
        return to_return

    def __sizeof__(self) -> int:
        size = object.__sizeof__(self)
        size += sum(sys.getsizeof(cf) for cf in self.CF_list)
        size += sum(child.__sizeof__() for child in self.children_list or [])
        return size
