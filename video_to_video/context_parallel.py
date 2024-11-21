import torch
import torch.distributed

_CONTEXT_PARALLEL_GROUP = None
_CONTEXT_PARALLEL_SIZE = None


def is_context_parallel_initialized():
    """
    Check if context parallel processing has been initialized.
    
    Returns:
        bool: True if context parallel group is initialized, False otherwise
    """
    if _CONTEXT_PARALLEL_GROUP is None:
        return False
    else:
        return True


def set_context_parallel_group(size, group):
    """
    Set the context parallel group and size manually.
    
    Args:
        size (int): Size of the context parallel group
        group: PyTorch distributed process group
    """
    global _CONTEXT_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_SIZE
    _CONTEXT_PARALLEL_GROUP = group
    _CONTEXT_PARALLEL_SIZE = size


def initialize_context_parallel(context_parallel_size):
    """
    Initialize context parallel processing by creating process groups.
    
    Creates distributed process groups of specified size, where each group
    contains consecutive ranks. Each process will be assigned to exactly one group.
    
    Args:
        context_parallel_size (int): Number of processes per context parallel group
        
    Raises:
        AssertionError: If context parallel group is already initialized
    """
    global _CONTEXT_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_SIZE

    assert _CONTEXT_PARALLEL_GROUP is None, "context parallel group is already initialized"
    _CONTEXT_PARALLEL_SIZE = context_parallel_size

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    for i in range(0, world_size, context_parallel_size):
        ranks = range(i, i + context_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _CONTEXT_PARALLEL_GROUP = group
            break


def get_context_parallel_group():
    """
    Get the current context parallel process group.
    
    Returns:
        torch.distributed.ProcessGroup: The current context parallel process group
        
    Raises:
        AssertionError: If context parallel group is not initialized
    """
    assert _CONTEXT_PARALLEL_GROUP is not None, "context parallel group is not initialized"

    return _CONTEXT_PARALLEL_GROUP


def get_context_parallel_world_size():
    """
    Get the size of the context parallel group.
    
    Returns:
        int: Number of processes in the context parallel group
        
    Raises:
        AssertionError: If context parallel size is not initialized
    """
    assert _CONTEXT_PARALLEL_SIZE is not None, "context parallel size is not initialized"

    return _CONTEXT_PARALLEL_SIZE


def get_context_parallel_rank():
    """
    Get the rank of the current process within its context parallel group.
    
    Returns:
        int: Rank of current process within its context parallel group (0 to size-1)
        
    Raises:
        AssertionError: If context parallel size is not initialized
    """
    assert _CONTEXT_PARALLEL_SIZE is not None, "context parallel size is not initialized"

    rank = torch.distributed.get_rank()
    cp_rank = rank % _CONTEXT_PARALLEL_SIZE
    return cp_rank


def get_context_parallel_group_rank():
    """
    Get the group index that the current process belongs to.
    
    Returns:
        int: Index of the context parallel group this process belongs to
        
    Raises:
        AssertionError: If context parallel size is not initialized
    """
    assert _CONTEXT_PARALLEL_SIZE is not None, "context parallel size is not initialized"

    rank = torch.distributed.get_rank()
    cp_group_rank = rank // _CONTEXT_PARALLEL_SIZE

    return cp_group_rank
