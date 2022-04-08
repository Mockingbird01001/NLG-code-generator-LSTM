
HOOKS = ['response']
def default_hooks():
    return {event: [] for event in HOOKS}
def dispatch_hook(key, hooks, hook_data, **kwargs):
    hooks = hooks or {}
    hooks = hooks.get(key)
    if hooks:
        if hasattr(hooks, '__call__'):
            hooks = [hooks]
        for hook in hooks:
            _hook_data = hook(hook_data, **kwargs)
            if _hook_data is not None:
                hook_data = _hook_data
    return hook_data
