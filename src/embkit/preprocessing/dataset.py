
def batch_iterable(iterable, batch_size):
    """
    Converts an iterable into a sequence of batched lists.

    Args:
        iterable: The input iterable (e.g., list, tuple, generator).
        batch_size: The desired size of each batch.

    Yields:
        list: A batch (list) of elements from the iterable.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    # Get an iterator from the iterable
    iterator = iter(iterable)

    # Loop indefinitely, or until StopIteration is raised
    while True:
        # Create a generator expression that yields up to batch_size elements
        # from the iterator. 'next(iterator, _SENTINEL)' attempts to get the 
        # next item; if it fails (StopIteration), it yields _SENTINEL.
        _SENTINEL = object()  # Unique sentinel value
        batch_gen = (next(iterator, _SENTINEL) for _ in range(batch_size))
        
        # Filter out the sentinel value(s) and create the list
        current_batch = [item for item in batch_gen if item is not _SENTINEL]

        # If the batch is empty, we've exhausted the iterable
        if not current_batch:
            return  # Stop the generator

        yield current_batch

