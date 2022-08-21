import lldb

def __lldb_init_module(debugger, dict):
    debugger.HandleCommand("type summary add -x \"Tensor\" -F debug_viewers.view_1d_tensor")


def view_1d_tensor(valobj, internal_dict, options):
    size = valobj.GetChildMemberWithName('size')