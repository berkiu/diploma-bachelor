import re
import numpy as np
from tqdm import tqdm


class XgbForest:
    def __init__(self, dataset, n_classes):
        ''' constructor '''

        self.xgb_trees = []
        self.dataset = dataset
        self.n_classes = n_classes

    def add_tree(self, tree):
        ''' function to add the tree '''

        self.xgb_trees.append(tree)

    def tree_gains(self, tree, dic):
        ''' function to get all gains from one tree '''

        dic[tree.node.feature].append(tree.node.gain)
        if not tree.left.node.is_leaf:
            self.tree_gains(tree.left, dic)
        if not tree.right.node.is_leaf:
            self.tree_gains(tree.right, dic)
        return dic

    def collect_gains(self, dic):
        ''' function to get all gains from forest '''

        for tree in self.xgb_trees:
            if not tree.node.is_leaf:
                dic = self.tree_gains(tree, dic)
        return dic

    def fit_trees(self):
        ''' function to count gains of nodes for all trees in forest '''

        for i in tqdm(range(len(self.xgb_trees))):
            tree = self.xgb_trees[i]
            if not tree.node.is_leaf:
                fitted_tree = tree.calculate_gains(self.dataset, tree)
                self.xgb_trees[i] = fitted_tree

    def predict(self, X):
        ''' function to predict new dataset '''

        predictions = np.array([self.predict_class(x) for x in X])
        return predictions

    def get_leaf_value(self, x, tree):
        ''' function to get leaf value for one sample '''

        if tree.node.is_leaf:
            return tree.node.leaf_value

        feature_ind = tree.node.feature_to_ind[tree.node.feature]
        if x[feature_ind] < tree.node.split_value:
            return self.get_leaf_value(x, tree.left)
        else:
            return self.get_leaf_value(x, tree.right)

    def predict_class(self, x):
        ''' function to predict class of sample '''

        leaf_values = []
        for tree in self.xgb_trees:
            leaf_values.append(self.get_leaf_value(x, tree))

        z_values = np.zeros(self.n_classes)
        for i in range(self.n_classes):
            exp_value = np.exp(np.sum(np.array(leaf_values[i::self.n_classes])))
            z_values[i] = exp_value

        p_output = z_values / np.sum(z_values)
        y_pred = int(np.where(p_output == np.amax(p_output))[0])
        return y_pred

    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''

        if tree.node.leaf_value is not None:
            print(tree.node.leaf_value)

        else:
            print(tree.node.feature, "<", tree.node.split_value, "gain:", tree.node.gain)
            print("%sleft:" % indent, end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % indent, end="")
            self.print_tree(tree.right, indent + indent)


class XgbTreeNode:
    def __init__(self, feature_to_ind, predict_class, n_classes):
        ''' constructor '''

        self.feature_to_ind = feature_to_ind
        self.predict_class = predict_class
        self.feature = ''
        self.gain = None
        self.number = -1
        self.left_child = None
        self.right_child = None
        self.leaf_value = None
        self.split_value = None
        self.is_leaf = False
        self.n_classes = n_classes


class XgbTree:
    def __init__(self, node):
        ''' constructor '''

        self.left = None
        self.right = None
        self.node = node

    def calculate_gains(self, dataset, tree):
        ''' function to calculate node gains '''

        dataset_left, dataset_right = self.split(dataset, tree.node.feature, tree.node.split_value,
                                                 tree.node.feature_to_ind)
        # check if childs are not null
        if len(dataset_left) > 0 and len(dataset_right) > 0:
            y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
            # compute information gain
            curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
            tree.node.gain = curr_info_gain
        else:
            tree.node.gain = 0.0
        if not tree.left.node.is_leaf:
            self.calculate_gains(dataset_left, tree.left)
        if not tree.right.node.is_leaf:
            self.calculate_gains(dataset_right, tree.right)
        return tree

    def entropy(self, y):
        ''' function to compute entropy '''

        count_classes = np.bincount(y.astype(int))
        if len(count_classes) < self.node.n_classes:
            add = np.zeros(self.node.n_classes - len(count_classes))
            count_classes = np.array(list(count_classes) + list(add))
        predicting_class_count = count_classes[self.node.predict_class]
        rest_class_count = len(y) - predicting_class_count
        ps = np.array([predicting_class_count / len(y), rest_class_count / len(y)])
        entropy = -np.sum([p * np.log2(p) for p in ps if p > 0])
        return entropy

    def gini_index(self, y):
        ''' function to compute gini index '''

        count_classes = np.bincount(y.astype(int))
        if len(count_classes) < self.node.n_classes:
            add = np.zeros(self.node.n_classes - len(count_classes))
            count_classes = np.array(list(count_classes) + list(add))
        predicting_class_count = count_classes[self.node.predict_class]
        rest_class_count = len(y) - predicting_class_count
        ps = np.array([predicting_class_count / len(y), rest_class_count / len(y)])
        gini = np.sum([p ** 2 for p in ps])
        return 1 - gini

    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        ''' function to compute information gain '''

        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode == "gini":
            gain = self.gini_index(parent) - (weight_l * self.gini_index(l_child) + weight_r * self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l * self.entropy(l_child) + weight_r * self.entropy(r_child))
        return gain

    def split(self, dataset, feature, threshold, feature_to_ind):
        ''' function to split the data '''

        feature = feature_to_ind[feature]
        dataset_left = np.array([row for row in dataset if row[feature] < threshold])
        dataset_right = np.array([row for row in dataset if row[feature] >= threshold])
        return dataset_left, dataset_right


class XgbModelParser:
    def __init__(self, n_classes):
        ''' constructor '''

        self.node_regex = re.compile(r'(\d+):\[(.*?)(?:<(.+)|)]\syes=(.*),no=(.*?),missing=.*')
        self.leaf_regex = re.compile(r'(\d+):leaf=(.*)')
        self.n_classes = n_classes

    def construct_xgb_tree(self, tree):
        ''' function to build a tree '''

        if tree.node.left_child is not None:
            tree.left = XgbTree(self.xgb_node_list[tree.node.left_child])
            self.construct_xgb_tree(tree.left)
        if tree.node.right_child is not None:
            tree.right = XgbTree(self.xgb_node_list[tree.node.right_child])
            self.construct_xgb_tree(tree.right)

    def parse_xgb_tree_node(self, line, feature_to_ind, predict_class):
        ''' function to parse lines from booster into nodes '''

        node = XgbTreeNode(feature_to_ind, predict_class, self.n_classes)

        m = self.leaf_regex.match(line)
        if m:
            node.number = int(m.group(1))
            node.leaf_value = float(m.group(2))
            node.is_leaf = True
        else:
            m = self.node_regex.match(line)
            node.number = int(m.group(1))
            node.feature = m.group(2)
            node.split_value = float(m.group(3))
            node.left_child = int(m.group(4))
            node.right_child = int(m.group(5))
            node.is_leaf = False
        return node

    def get_xgb_model_from_memory(self, dump, max_trees, dataset, feature_to_ind):
        ''' function to create Forest from xgb boosters description '''

        forest = XgbForest(dataset, self.n_classes)
        self.xgb_node_list = {}
        num_tree = 0
        for i in tqdm(range(len(dump))):
            booster_line = dump[i]
            self.xgb_node_list = {}
            for line in booster_line.split('\n'):
                line = line.strip()
                if not line:
                    continue
                node = self.parse_xgb_tree_node(line, feature_to_ind, num_tree % self.n_classes)
                if not node:
                    return None
                self.xgb_node_list[node.number] = node
            num_tree += 1
            tree = XgbTree(self.xgb_node_list[0])
            self.construct_xgb_tree(tree)
            forest.add_tree(tree)
            if num_tree == max_trees:
                break
        return forest
