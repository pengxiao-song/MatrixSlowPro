from .graph import default_graph

def get_node_from_graph(node_name, name_scope=None, graph=None):
    if graph is None:
        graph = default_graph
    if name_scope:
        node_name = name_scope + '/' + node_name
    for node in graph.nodes:
        if node.name == node_name:
            return node
    return None

class name_scope:
    def __init__(self, name_scope):
        self.name_scope = name_scope
        
    def __enter__(self):
        default_graph.name_scope = self.name_scope
        
    def __exit__(self, exc_type, exc_value, exc_tb):
        default_graph.name_scope = None