import numpy as np


class Graph:
    '''
    计算图类
    '''
    def __init__(self, name_scope=None):
        self.nodes = []
        self.name_scope = name_scope

    def add_node(self, node):
        '''
        添加节点
        '''
        self.nodes.append(node)

    def clear_jacobi(self):
        '''
        清除计算图中所有节点的雅可比矩阵
        '''
        for node in self.nodes:
            node.clear_jacobi()

    def reset_value(self):
        '''
        重置计算图中所有节点的值
        '''
        for node in self.nodes:
            node.reset_value(recursive=False)  # 不递归清除子节点的值

    def count_node(self):
        return len(self.nodes)

    def draw(self, ax=None):
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
        except:
            raise ModuleNotFoundError("Need Module: 'networkx' and 'matplotlib'")

        G = nx.Graph()

        already = []
        labels = {}
        for node in self.nodes:
            G.add_node(node)
            labels[node] = node.name + ("({:s})".format(str(node.dim)) if hasattr(node, "dim") else "") \
                + ("\n[{:.3f}]".format(np.linalg.norm(node.jacobi))
                   if node.jacobi is not None else "")

            for c in node.get_children():
                if (node, c) not in already:
                    G.add_edge(node, c)
                    already.append((node, c))

            for p in node.get_parents():
                if (p, node) not in already:
                    G.add_edge(p, node)
                    already.append((p, node))

        if ax is None:
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111)

        ax.clear()
        ax.axis("on")
        ax.grid(True)

        pos = nx.spring_layout(G, seed=42)

        # 有雅克比的变量节点
        cm = plt.cm.Reds
        nodelist = [n for n in self.nodes if n.__class__.__name__ ==
                    "Variable" and n.jacobi is not None]
        colorlist = [np.linalg.norm(n.jacobi) for n in nodelist]
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color=colorlist, cmap=cm, edgecolors="#666666",
                               node_size=2000, alpha=1.0, ax=ax)

        # 无雅克比的变量节点
        nodelist = [n for n in self.nodes if n.__class__.__name__ ==
                    "Variable" and n.jacobi is None]
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color="#999999", cmap=cm, edgecolors="#666666",
                               node_size=2000, alpha=1.0, ax=ax)

        # 有雅克比的计算节点
        nodelist = [n for n in self.nodes if n.__class__.__name__ !=
                    "Variable" and n.jacobi is not None]
        colorlist = [np.linalg.norm(n.jacobi) for n in nodelist]
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color=colorlist, cmap=cm, edgecolors="#666666",
                               node_size=2000, alpha=1.0, ax=ax)

        # 无雅克比的计算节点
        nodelist = [n for n in self.nodes if n.__class__.__name__ !=
                    "Variable" and n.jacobi is None]
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color="#999999", cmap=cm, edgecolors="#666666",
                               node_size=2000, alpha=1.0, ax=ax)

        nx.draw_networkx_edges(G, pos, width=2, edge_color="#014b66", ax=ax)
        nx.draw_networkx_labels(G, pos, labels=labels, font_weight="bold", font_color="#6c6c6c", font_size=8,
                                font_family='arial', ax=ax)


default_graph = Graph()
