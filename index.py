import random, time, sys
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
import pickle


class Node:
    def __init__(self, K=None, parent=None):
        assert K is not None or parent, "Either `K` should be provided for root node, or `parent` for internal nodes"
        # Reference to parent node. Used in ANNS search
        self.parent = parent
        # depth start from 0. To compute dimension, relevant to the level, use (self.depth % self.K)
        self.depth = (parent.depth + 1) if parent else 0
        # K means number of vector dimensions
        self.K = parent.K if parent else K
        # value, which splits subspace into to parts using hyperplane: item[self.depth % self.K] == self.pivot
        # pivot is empty for any leaf node.
        self.pivot = None
        # left and right child nodes
        self.left = None
        self.right = None
        # collection of items
        self.items = None

    def build_kd_tree(self, items, leaf_capacity=4):
        '''Takes a list of items and arranges it in a kd-tree'''
        assert items is not None, "Please provide at least one point"
        # put all items in the node if they fit into limit
        if len(items) <= leaf_capacity:
            self.items = items
        # or else split items into 2 subnodes using median value
        else:
            self.items = None
            self.left = Node(parent=self)
            self.right = Node(parent=self)

            # TODO 1.A.: fill in the code to initialize internal node.
            # Be careful: there may be multiple items with the same values as pivot,
            # make sure they go to the same child.
            # Also, there may be duplicate items, and you need to deal with them
            self.pivot = None  # here you should write median value with respect to coordinate
            left = None  # those items, which are smaller than the pivot value
            right = None  # those items, which are greater than the pivot value

            p_sorted = sorted(items, key=lambda p: p[0][self.depth % self.K])
            # if all values of a current dimension are the same, check if elements are actually the same
            if p_sorted[0][0][self.depth % self.K] == p_sorted[len(p_sorted) - 1][0][self.depth % self.K]:
                all_same = True
                for i in range(len(p_sorted) - 1):
                    # compare vectors
                    if not np.array_equal(p_sorted[i][0], p_sorted[i + 1][0]):
                        all_same = False
                        break
                if all_same:
                    # make it a leaf node
                    self.items = items
                    self.left = None
                    self.right = None
                    return self

            # find median and assign its value to a pivot
            med = len(p_sorted) // 2
            self.pivot = p_sorted[med][0][self.depth % self.K]

            # set median id to the first occurence of median value
            while med > 0 and p_sorted[med - 1][0][self.depth % self.K] == self.pivot:
                med -= 1

                # move pivot to the right in case if all elements in the beginning have the same value as pivot
            if med == 0 and self.pivot != p_sorted[len(p_sorted) - 1][0][self.depth % self.K]:
                med = len(p_sorted) // 2
                while p_sorted[med][0][self.depth % self.K] == self.pivot:
                    med += 1
                self.pivot = p_sorted[med][0][self.depth % self.K]

            left, right = p_sorted[:med], p_sorted[med:]

            self.left.build_kd_tree(left)
            self.right.build_kd_tree(right)

        return self

    def kd_find_leaf(self, key):
        ''' returns a node where key should be stored (but can be not present)'''
        if self.pivot is None or self.items is not None:  # leaf node OR empty root
            return self
        else:

            # TODO 1.B. This is a basic operation for travesing the tree.
            # define correct path to continue recursion
            if key[self.depth % self.K] <= self.pivot:
                return self.left.kd_find_leaf(key)
            else:
                return self.right.kd_find_leaf(key)

    #     def kd_insert_no_split(self, item):
    #         '''Naive implementation of insert into leaf node. It is not used in tests of this tutorial.'''
    #         node = self.kd_find_leaf(item[0])
    #         node.items.append(item)

    def kd_insert_with_split(self, item, leaf_capacity=4):
        '''This method recursively splits the nodes into 2 child nodes if they overflow `leaf_capacity`'''

        # TODO 1.C. This is very simple insertion procedure.
        # Split the node if it cannot accept one more item.
        # HINT: reuse kd_find_leaf() and build_kd_tree() methods if possible

        node = self.kd_find_leaf(item[0])
        node.build_kd_tree((node.items or []) + [item], leaf_capacity)

    def get_subtree_items(self):
        '''Returns union of all items belonging to a subtree'''
        if self.pivot is None or self.items is not None:  # leaf node OR empty root
            return self.items
        else:
            return self.left.get_subtree_items() + self.right.get_subtree_items()

    def get_nn(self, key, knn):
        '''Return K approximate nearest neighbours for a given key'''
        node = self.kd_find_leaf(key)
        best = []

        # TODO 1.D. ANN search.
        # write here the code which returns `knn`
        # approximate nearest neighbours with respect to euclidean distance
        # basically, you need to move up through the parents chain until the number of elements
        # in a parent subtree is more or equal too the expected number of nearest neighbors,
        # and then return top-k elements of this subtree sorted by euclidean distance
        # HINT: you can use [scipy.spatial.]distance.euclidean(a, c) - it is already imported

        while node is not None and len(best) < knn:
            best = node.get_subtree_items()
            node = node.parent
        best = sorted(best, key=lambda p: distance.euclidean(p[0], key))

        return best[:knn]

    def get_in_range(self, lower_left_bound_key, upper_right_bound_key):
        '''Runs range query. Returns all items bounded by the given corners: `lower_left_bound_key`, `upper_right_bound_key`'''
        result = []
        if self.pivot is None or self.items is not None:  # internal node OR empty root
            # TODO 3.B.: This is a leaf node. Select only those items from self.item
            # which fall into a given range
            for item in self.items:
                _in = [low <= v <= up for low, v, up in zip(lower_left_bound_key, item[0], upper_right_bound_key)]
                inside = all(_in)
                if inside:
                    result.append(item)
            return result
        else:
            # TODO 3.B.: This is an internal node.
            # write recursive code to collect corresponding data from subtrees
            # compare pivot to the bounds to decide whether to consider any of its children for search

            skip_right = self.pivot > upper_right_bound_key[self.depth % self.K]
            skip_left = self.pivot < lower_left_bound_key[self.depth % self.K]
            result = [] if skip_left else self.left.get_in_range(lower_left_bound_key, upper_right_bound_key)
            result += [] if skip_right else self.right.get_in_range(lower_left_bound_key, upper_right_bound_key)
            return result
