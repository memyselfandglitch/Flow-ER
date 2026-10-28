import unittest
from utils.inference_utils import sample_chunk_size, collect_sample_chunks


class SimulatedOOM(Exception):
    pass


class SampleChunkTests(unittest.TestCase):
    def test_failing_394_atom_case_is_split(self):
        self.assertEqual(sample_chunk_size(1, 394, 64), 16)
        self.assertEqual(sample_chunk_size(1, 133, 64), 64)
        self.assertLessEqual(sample_chunk_size(3, 394, 64) * 3 * 394**2, 4_000_000)

    def test_retry_keeps_completed_chunks_and_all_64_samples(self):
        calls, releases, retries = [], [], []
        def run(size):
            calls.append(size)
            if len(calls) == 2:
                raise SimulatedOOM()
            return len(calls)
        chunks = collect_sample_chunks(64, 32, run, SimulatedOOM,
                                       lambda: releases.append(True),
                                       lambda a, b: retries.append((a, b)))
        self.assertEqual(calls, [32, 32, 16, 16])
        self.assertEqual(chunks, [(1, 32), (3, 16), (4, 16)])
        self.assertEqual(sum(size for _, size in chunks), 64)
        self.assertEqual(len(releases), 1)
        self.assertEqual(retries, [(32, 16)])

    def test_uneven_sample_count_is_not_dropped(self):
        chunks = collect_sample_chunks(7, 4, lambda n: n, SimulatedOOM, lambda: None, lambda *x: None)
        self.assertEqual(chunks, [(4, 4), (3, 3)])

    def test_one_sample_oom_and_non_memory_errors_propagate(self):
        def oom(n): raise SimulatedOOM('still too large')
        with self.assertRaises(SimulatedOOM):
            collect_sample_chunks(64, 1, oom, SimulatedOOM, lambda: None, lambda *x: None)
        def other(n): raise ValueError('not a memory error')
        with self.assertRaises(ValueError):
            collect_sample_chunks(64, 16, other, SimulatedOOM, lambda: None, lambda *x: None)


if __name__ == '__main__':unittest.main()
