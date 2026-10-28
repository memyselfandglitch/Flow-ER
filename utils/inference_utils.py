"""Memory-bounded sample scheduling without changing samples per reaction."""


def sample_chunk_size(batch_size, atoms, samples, max_matrix_elements=4_000_000,
                      max_samples=None):
    if min(batch_size, atoms, samples, max_matrix_elements) < 1:
        raise ValueError('Batch, atom, sample, and matrix limits must be positive')
    if max_samples is not None and max_samples < 1:
        raise ValueError('Sample chunk limit must be positive')
    limit = min(samples, max_samples or samples,
                max(1, max_matrix_elements // (batch_size * atoms * atoms)))
    # Powers of two make the usual 64-sample run evenly chunked.
    return 1 << (limit.bit_length() - 1)


def collect_sample_chunks(samples, chunk_size, run_chunk, oom_type,
                          release_memory, report_retry):
    """Retry only a failed chunk, after its exception frame releases tensors.

    run_chunk returns a CPU result. Successful chunks are never recomputed.
    A single-sample OOM remains an explicit error rather than dropping a reaction.
    """
    if samples < 1 or chunk_size < 1:
        raise ValueError('Sample count and chunk size must be positive')
    completed = []
    remaining = samples
    while remaining:
        size = min(chunk_size, remaining)
        failed = False
        try:
            result = run_chunk(size)
        except oom_type:
            if size == 1:
                raise
            failed = True
        if failed:
            # Do this outside except: the traceback otherwise keeps GPU tensors live.
            release_memory()
            chunk_size = max(1, size // 2)
            report_retry(size, chunk_size)
            continue
        completed.append((result, size))
        remaining -= size
    return completed
