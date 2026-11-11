"""Small stand-ins for `keyboard` and `msgpackrpc`, used only by the tests."""
import types

# keyboard: only keyboard.wait() is used (keyboard_debug mode)
keyboard = types.ModuleType("keyboard")
keyboard.wait = lambda *args, **kwargs: None

# msgpackrpc: only msgpackrpc.error.RPCError is used (camera retry logic)
msgpackrpc = types.ModuleType("msgpackrpc")
msgpackrpc_error = types.ModuleType("msgpackrpc.error")


class RPCError(Exception):
    pass


msgpackrpc_error.RPCError = RPCError
msgpackrpc.error = msgpackrpc_error
