import contextlib
import os


def action_fcurves(action, id_type="OBJECT"):
    """Return curves for an importer action (one slot per action).

    Keep the legacy API on 4.5 for exporter compatibility. Blender 5 uses
    a slot and channelbag instead. Light actions must initialize with LIGHT;
    node tree actions created by keyframe_insert already have the correct slot.
    """
    if hasattr(action, "fcurves"):
        return action.fcurves
    from bpy_extras.anim_utils import action_ensure_channelbag_for_slot

    if len(action.slots) > 1:
        raise ValueError("Expected a single-slot importer action: " + action.name)
    slot = action.slots[0] if action.slots else action.slots.new(id_type, action.name)
    return action_ensure_channelbag_for_slot(action, slot).fcurves


def new_grouped_fcurve(action, data_path, index, action_group):
    """Create a bone curve with its group using either animation API."""
    curves = action_fcurves(action)
    if hasattr(action, "fcurves"):
        return curves.new(data_path, index=index, action_group=action_group)
    return curves.new(data_path, index=index, group_name=action_group)


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
            inspectPrefix = " ┃ " if node.children else "   "
            inspector(node, prefix + inspectPrefix)
        for child in node.children:
            _printNode(child, prefix, child is node.children[-1])

    _printNode(root)
