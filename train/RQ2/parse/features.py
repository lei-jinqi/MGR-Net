import hashlib
import networkx as nx
from typing import List, Dict


class WeisfeilerLehmanHashing(object):
    """
    Weisfeiler-Lehman feature extractor class.

    Args:
        graph (NetworkX graph): NetworkX graph for which we do WL hashing.
        wl_iterations (int): Number of WL iterations.
        attributed (bool): Presence of attributes.
        erase_base_feature (bool): Deleting the base features.
    """

    def __init__(
        self,
        graph: nx.classes.graph.Graph,
        wl_iterations: int,
        attributed: bool,
        erase_base_features: bool,
        use_path: bool
    ):
        """
        Initialization method which also executes feature extraction.
        """
        self.wl_iterations = wl_iterations
        self.graph = graph
        self.attributed = attributed
        self.erase_base_features = erase_base_features
        self._set_features()
        self._do_recursions()

        self.use_degree = True

        self.use_path = use_path
        self.get_path_features()

    def _set_features(self):
        """
        Creating the features.
        """
        if self.attributed:
            self.features = nx.get_node_attributes(self.graph, "feature")
        else:
            self.features = {
                node: self.graph.degree(node) for node in self.graph.nodes()
            }
        self.extracted_features = {k: [str(v)] for k, v in self.features.items()}
        

    def _erase_base_features(self):
        """
        Erasing the base features
        """
        for k, v in self.extracted_features.items():
            del self.extracted_features[k][0]

    def _do_a_recursion(self):
        """
        The method does a single WL recursion.

        Return types:
            * **new_features** *(dict of strings)* - The hash table with extracted WL features.
        """
        new_features = {}
        for node in self.graph.nodes():
            nebs = self.graph.neighbors(node)
            degs = [self.features[neb] for neb in nebs]
            features = [str(self.features[node])] + sorted([str(deg) for deg in degs])
            features = "_".join(features)
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing
        self.extracted_features = {
            k: self.extracted_features[k] + [v] for k, v in new_features.items()
        }
        return new_features

    def _do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.wl_iterations):
            self.features = self._do_a_recursion()
        if self.erase_base_features:
            self._erase_base_features()

    def get_node_features(self) -> Dict[int, List[str]]:
        """
        Return the node level features.
        """
        return self.extracted_features

    def get_graph_features(self) -> List[str]:
        """
        Return the graph level features.
        """
        if self.use_path:
            return [
                feature
                for node, features in self.extracted_features.items()
                for feature in features
            ] + self.path_features
        else:
            return [
                feature
                for node, features in self.extracted_features.items()
                for feature in features
            ]
    
    def _dfs(self,node,visited,v_node,paths):
        visited.append(node)
        v_node[node] += 1
        if self.graph.out_degree(node) == 0:
            paths.append(visited.copy())
            return
        else:
            for i in self.graph.successors(node):
                if i not in visited and v_node[node] <= len(self.graph):
                    self._dfs(i,visited,v_node,paths)
        visited.pop()

    def get_path_features(self) -> List[str]:
        if len(self.graph) ==0:
            self.path_features =[]
            return 
        paths = []
        node = list(self.graph.nodes)[0]
        v_node = {}
        for i in self.graph.nodes:
            v_node[i] = 0
        self._dfs(node,[],v_node,paths)
        if self.use_degree:
            paths = self._change_to_degree(paths)
        
        features = []
        for i in paths:
            p = [str(num) for num in i]
            f = "_".join(p)
            hash_object = hashlib.md5(f.encode())
            hashing = hash_object.hexdigest()
            features.append(hashing)
        self.path_features = features

    def _change_to_degree(self, paths):
        res = []
        for path in paths:
            newpath = []
            for node in path:
                newpath.append(self.graph.degree(node))
            res.append(newpath)
        return res
        