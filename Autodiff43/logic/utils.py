"""
Utility functions
"""

def topological_sort(root_node):
    """
    Finds the topological sort of a graph given the root node
    Args:
        The root node that should be the end of the topological sort
    Returns:
        Because we are reversing, it returns a list with the reverse topological sort (the root note is first)
    """
    visited = set()

    def dfs(node, topo_sort):
        nonlocal visited

        if (node in visited):
            return topo_sort
        visited.add(node)

        for (nei, _) in node.node_edges:
            topo_sort = dfs(nei, topo_sort)
        
        return topo_sort + [node]

    return reversed(dfs(root_node, []))

def clear_grad(root_node):
    """
    Clears the gradients of a graph given the root node
    Args:
        The root node from where all the gradients are cleared
    Returns:
        None
    """
    visited = set()

    def dfs(node):
        nonlocal visited

        if (node in visited):
            return
        visited.add(node)

        node.grad = 0

        for (child, _) in node.node_edges:
            dfs(child)

    dfs(root_node)