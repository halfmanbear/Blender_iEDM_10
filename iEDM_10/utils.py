

import contextlib, os

@contextlib.contextmanager
def chdir(to):
    original = os.getcwd()
    try: 
      os.chdir(to)
      yield
    finally:
      os.chdir(original)

def get_root_object(obj):
  """Given an object, returns the root node.
  Follows 'parent' attribute references until none remain."""
  while obj.parent:
    obj = obj.parent
  return obj


def print_edm_graph(root, inspector=None):
  """Prints a graph of the tree, optionally with an inspection function"""

  def _printNode(node, prefix=None, last=True):
    if prefix is None:
      firstPre = ""
      prefix = ""
    else:
      firstPre = prefix + (" ┗━" if last else " ┣━")
      prefix = prefix + ("   " if last else " ┃ ")
    print(firstPre + repr(node))
    if inspector is not None:
      inspectPrefix = (" ┃ " if node.children else "   ")
      inspector(node, prefix + inspectPrefix)
    for child in node.children:
      _printNode(child, prefix, child is node.children[-1])

  _printNode(root)
