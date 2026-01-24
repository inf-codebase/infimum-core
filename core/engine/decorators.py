def singleton(cls):
    """Decorate for singleton class
    example:
    @singleton
    class SingletonClass:
        def __init__(self):
            self.value = None
        
        def some_method(self):
            pass
    
    Supports parameterized singletons - different instances for different arguments.
    """
    instances = {}
    def get_instance(*args, **kwargs):
        # Create cache key from class and arguments
        # Use args and sorted kwargs for consistent hashing
        key = (cls, args, tuple(sorted(kwargs.items())))
        if key not in instances:
            instances[key] = cls(*args, **kwargs)
        return instances[key]
    return get_instance
