"""Regression checks for the confounds targeted by reviewer controls."""
import unittest
import numpy as np
from model import model as p
from model.Paper.reviewer_experiments import permute_time


class ReviewerControlTests(unittest.TestCase):
    def summary_fixture(self):
        names=p.FULL_FEATURE_NAMES
        x=np.zeros((1,11,len(names)),dtype=np.float32)
        positions={name:i for i,name in enumerate(names)}
        x[0,:,positions['elapsed_time']]=np.arange(11)/10
        x[0,1:,positions['delta_time']]=.1
        x[0,:,positions['speed']]=[0,.05,.1,1,2,3,4,5,6,7,8]
        output_names=p.classical_summary_feature_names(names)
        return x,positions,output_names

    def test_pause_summary_uses_within_movement_speed_percentile(self):
        x,pos,names=self.summary_fixture()
        actual=p.summarize_sequences(x,p.FULL_FEATURE_NAMES)
        # The 10th percentile is .05: first two samples qualify, but the
        # first sample has zero elapsed interval. This is not a fixed-speed
        # inactivity detector.
        self.assertAlmostEqual(actual[0,names.index('derived:pause_duration')],.1)

    def test_click_summary_uses_first_positive_including_fractional_state(self):
        x,pos,names=self.summary_fixture()
        x[0,4,pos['button_state']]=.2
        x[0,5,pos['button_state']]=1
        actual=p.summarize_sequences(x,p.FULL_FEATURE_NAMES)
        self.assertAlmostEqual(actual[0,names.index('derived:click_timing')],.4)

    def test_click_summary_without_positive_state_falls_back_to_duration(self):
        x,pos,names=self.summary_fixture()
        actual=p.summarize_sequences(x,p.FULL_FEATURE_NAMES)
        self.assertAlmostEqual(actual[0,names.index('derived:click_timing')],1.)

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
