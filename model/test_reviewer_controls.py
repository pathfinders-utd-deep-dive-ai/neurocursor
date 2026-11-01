"""Regression checks for the confounds targeted by reviewer controls."""
import unittest
import numpy as np
from model import model as p
from model.Paper.reviewer_experiments import permute_time


class ReviewerControlTests(unittest.TestCase):
    def test_experimental_inputs_cannot_export_incompatible_manifest(self):
        for options in ({'sequence_transform':lambda x:x},
                        {'classical_distribution_only':True}):
            with self.assertRaisesRegex(ValueError,'cannot be exported'):
                p.run_configuration(None,None,None,None,'cnn-gru','full',
                                    True,False,**options)

    def test_permutation_keeps_points_but_changes_order(self):
        x=np.random.default_rng(17).normal(size=(4,128,2)).astype('float32')
        y=permute_time(x,31)
        self.assertFalse(np.array_equal(x,y))
        for a,b in zip(x,y):
            self.assertEqual(set(map(tuple,a)),set(map(tuple,b)))
        np.testing.assert_allclose(p.summarize_sequences(x,None),
            p.summarize_sequences(y,None),rtol=1e-6,atol=1e-6)

    def test_reversal_keeps_distributional_baseline_inputs(self):
        x=np.random.default_rng(20).normal(size=(4,128,2)).astype('float32')
        np.testing.assert_allclose(p.summarize_sequences(x,None),
            p.summarize_sequences(x[:,::-1,:],None),rtol=1e-6,atol=1e-6)

    def test_channel_removal_preserves_shared_summary_values(self):
        x=np.random.default_rng(24).normal(size=(4,128,27)).astype('float32')
        kept=[i for i,f in enumerate(p.FULL_FEATURE_NAMES)
              if not any(t in f for t in
                  ('acceleration','jerk','angular_velocity','curvature'))]
        full=p.summarize_sequences(x,None)
        reduced=p.summarize_sequences(x[:,:,kept],None)
        # Retained feature values are identical; only the designated channel
        # summaries disappear. Neither arm gains the 13 derived statistics.
        indexes=[stat*27+i for stat in range(4) for i in kept]
        np.testing.assert_allclose(full[:,indexes],reduced,
                                   rtol=1e-6,atol=1e-6)
        self.assertEqual(full.shape[1],108)
        self.assertEqual(reduced.shape[1],76)


if __name__=='__main__':unittest.main()
