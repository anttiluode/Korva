"""The organ alone, numpy only: does the comb find the period, and is its echo the last lap?"""
import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import organ as O


class Organ(unittest.TestCase):
    def test_best_ring_is_the_period_or_a_multiple(self):
        rng = np.random.default_rng(5)
        hits = 0
        for _ in range(30):
            x, clean, P, ch = O.song(rng, T=256, p_drop=0.0, p_insert=0.0, change=False, p_fill=0.0)
            _, laps, _ = O.organ(x)
            hits += laps[-1, 0] % P == 0
        self.assertGreaterEqual(hits, 27)

    def test_echo_of_the_period_ring_is_the_last_lap(self):
        rng = np.random.default_rng(6)
        x, clean, P, ch = O.song(rng, T=256, p_drop=0.0, p_insert=0.0, change=False, p_fill=0.0)
        tok, laps, _ = O.organ(x)
        for t in range(3 * P, 255):
            if laps[t, 0] == P:
                np.testing.assert_array_equal(tok[t, 0, :8], x[t + 1 - P])

    def test_gated_ring_keeps_a_dropped_beat(self):
        """With gating, one dropped beat does not overwrite the ring (the slow surprise stays low)."""
        rng = np.random.default_rng(7)
        x, clean, P, ch = O.song(rng, T=400, p_drop=0.0, p_insert=0.0, change=False, p_fill=0.0)
        t0 = next(t for t in range(4 * P, 400 - P) if x[t].any())
        y = x.copy(); y[t0] = 0
        t = t0 + P - 1                       # one lap later, predicting tick t0 + P
        for gated, kept in ((True, True), (False, False)):
            tok, laps, _ = O.organ(y, gated=gated)
            self.assertEqual(laps[t, 0] % P, 0)
            self.assertEqual(bool((tok[t, 0, :8] == x[t0]).all()), kept, gated)


if __name__ == "__main__":
    unittest.main()
