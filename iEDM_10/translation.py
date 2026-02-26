"""
translation.py

Holds the translation tree that we use to step to/from blender/edm forms
"""

from inspect import isgenerator

from .utils import get_root_object


# Prefixes for render node categories.
_prefixLookup = {
    "transform": "tf",
    "CONNECTORS": "cn",
    "RENDER_NODES": "rn",
    "SHELL_NODES": "shell",
    "LIGHT_NODES": "light",
}


class TranslationNode(object):
    """Holds a triple of blender object, render node and transform nodes.

    Each TranslationNode maps to maximum ONE blender object, maximum ONE renderNode
    and maximum ONE transform node. It may map to more than one type, in cases
    where they are directly equatable.
    """

    def __init__(self, blender=None, render=None, transform=None):
        self.blender = blender
        self.render = render
        self.transform = transform

        self.parent = None
        self.graph = None
        self.children = []

    @property
    def name(self):
        if self.blender:
            return "bl:" + self.blender.name

        elif self.render and self.render.name:
            return (
                _prefixLookup[self.render.category.value]
                + ":"
                + self.render.name
            )

        elif self.transform and self.transform.name:
            return "tf:" + self.transform.name

        else:
            parts = []
            if self.render:
                parts.append(_prefixLookup[self.render.category.value])
            if self.transform:
                parts.append("tf")
            return "Unnamed" + (" " if parts else "") + "/".join(parts)

    def insert_parent(self):
        return self.graph.insert_new_parent(self)

    @property
    def type(self):
        "Easy test to see if we are a simple node"
        if self.blender and not self.render and not self.transform:
            return "BLEND"
        if not self.blender and self.render and not self.transform:
            return "RENDER"
        if not self.blender and not self.render and self.transform:
            return "TRANSFORM"
        if any([self.blender, self.render, self.transform]):
            return "MIXED"
        return None


class RootTranslationNode(TranslationNode):
    """Acts as the root node of a translation graph"""

    def __init__(self):
        # Root is always empty, but keeps tree structure.
        self.transform = None
        self.render = None
        self.blender = None
        self.children = []
        self.parent = None
        self.graph = None

    @property
    def name(self):
        return "<ROOT>"


class TranslationGraph(object):
    def __init__(self):
        self.nodes = [RootTranslationNode()]
        self.root = self.nodes[0]
        self.root.graph = self

    def print_tree(self, inspector=None):
        """Prints a graph of the tree, optionally with an inspection function"""

        def _printNode(node, prefix=None, last=True):
            if prefix is None:
                firstPre = ""
                prefix = ""
            else:
                firstPre = prefix + (" ┗━" if last else " ┣━")
                prefix = prefix + ("   " if last else " ┃ ")

            print(
                firstPre
                + node.name.ljust(30 - len(firstPre))
                + " Render: "
                + str(node.render).ljust(30)
                + " Trans: "
                + str(node.transform)
            )

            if inspector is not None:
                inspectPrefix = (" ┃ " if node.children else "   ")
                inspector(node, prefix + inspectPrefix)

            for child in node.children:
                _printNode(child, prefix, child is node.children[-1])

        if self.root:
            _printNode(self.root)

    def walk_tree(self, walker, include_root=True):
        """Accepts a function, and calls it for every node in the tree depth first.

        The parent is guaranteed to be initialised before the child. Any changes
        to the collection of children of the active node are respected, and the
        children will be walked. It is also safe to insert additional parents
        between the original and the current node. Anything else is undefined.

        If the walker function is a generator, it will be called twice; once
        before, and once after the children are walked.
        """

        def _walk_node(node):
            ret = walker(node)
            if isgenerator(ret):
                try:
                    next(ret)
                except StopIteration:
                    # Is a generator, but only one return value for this node
                    ret = None

            for child in list(node.children):
                _walk_node(child)

            # If this was a generator, we need to call again but this must be the last time
            if isgenerator(ret):
                try:
                    next(ret)
                except StopIteration:
                    pass
                else:
                    raise RuntimeError(
                        "Tree walker generator did not terminate on second call"
                    )

        if include_root:
            _walk_node(self.root)
        else:
            for root in list(self.root.children):
                _walk_node(root)

    def attach_node(self, node, parent):
        """Adds a new child to a parent node"""
        assert parent in self.nodes, "Parent must exist in node graph"
        assert node not in self.nodes, "Attempting to reattach child already in graph"
        assert not node.children, "New child must not have children"

        node.graph = self
        self.nodes.append(node)
        parent.children.append(node)
        node.parent = parent

    def remove_node(self, node):
        assert node in self.nodes, "Node not in graph"
        assert node.parent, "Invalid node: No parent. Cannot remove root node."

        parent = node.parent
        # Preserve sibling ordering by splicing children at removed node index.
        insert_at = parent.children.index(node)
        parent.children.pop(insert_at)

        for child in list(node.children):
            child.parent = parent
            parent.children.insert(insert_at, child)
            insert_at += 1

        node.children = []
        node.parent = None
        node.graph = None
        self.nodes.remove(node)

    def insert_new_parent(self, node):
        """Inserts a new node between the passed argument node and its parent"""
        assert node in self.nodes, "Node not in graph"
        assert node is not self.root, "Cannot insert above root"

        # Create the new node attached to the old parent
        parent = node.parent
        index = parent.children.index(node)
        newNode = TranslationNode()
        self.attach_node(newNode, parent)
        parent.children.remove(newNode)
        parent.children.insert(index, newNode)

        # Move the node off its original parent onto the new node
        parent.children.remove(node)
        newNode.children.append(node)
        node.parent = newNode

        return newNode

    # Helpful, but not necessarily related directly to reading or writing,
    # construction helpers

    def get_all_render_nodes(self):
        """Returns a list of all render nodes in the graph"""
        return [n.render for n in self.nodes if n.render]

    @classmethod
    def from_blender_objects(cls, blender_objects):
        """Takes a list of blender objects and builds the basic translation graph"""
        graph = cls()

        # Walk up the graphs to get all 'root' objects
        roots = set(get_root_object(x) for x in blender_objects)

        nodeObjectMap = {}

        def _create_node(obj):
            """Creates a graph node from a blender object"""
            node = TranslationNode(blender=obj)
            nodeObjectMap[obj] = node

            if not obj.parent:
                parent = graph.root
            else:
                parent = nodeObjectMap[obj.parent]

            graph.attach_node(node, parent)

            for child in obj.children:
                _create_node(child)

        for root in roots:
            _create_node(root)

        return graph
