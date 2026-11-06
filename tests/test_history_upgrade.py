"""Checks for deduplication, temporal boundaries, masking, and independent seeds."""
import unittest
import argparse
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import torch
import history_upgrade as upgrade
from history_upgrade import unique_clicks,long_history,other_click_context,SharedHistoryEnsemble


class HistoryUpgradeTests(unittest.TestCase):
    def setUp(self):
        # Synthetic checkpoint tests must not require competition-data caches.
        fingerprint = patch.object(upgrade.common, 'cache_fingerprint', lambda: 'synthetic-test-data')
        fingerprint.start()
        self.addCleanup(fingerprint.stop)
        self.data={'user':np.array([0,0,1]),'time':np.array([100,200,250]),
            'items':np.tile(np.arange(10),(3,1)),
            'hist':np.array([[0,1]+[-1]*18,[0,1,0]+[-1]*17,[2]+[-1]*19]),
            'hist_time':np.array([[20,30]+[0]*18,[20,30,100]+[0]*17,[40]+[0]*19]),
            'label':np.array([[1]+[0]*9,[0,1]+[0]*8,[0]*10])}
        self.clicks=unique_clicks(self.data,np.array([0,1]))

    def test_dedup_preserves_repeated_clicks_at_different_times(self):
        self.assertEqual(len(self.clicks),4)
        self.assertEqual(int((self.clicks['item']==1).sum()),2)

    def test_reconstructed_history_excludes_event_and_future_clicks(self):
        h,n=long_history(self.data,np.array([0,1]),self.clicks,11,limit=10)
        self.assertEqual(n.tolist(),[2,3])
        np.testing.assert_array_equal(h[0,:2],[2,1])
        np.testing.assert_array_equal(h[1,:3],[1,2,1])

    def test_own_query_history_survives_unknown_user(self):
        h,n=long_history(self.data,np.array([2]),self.clicks,11,limit=10)
        self.assertEqual(n.tolist(),[1])
        self.assertEqual(h[0,0],3)

    def test_query_labels_have_no_effect(self):
        before=long_history(self.data,np.array([2]),self.clicks,11)
        changed=dict(self.data,label=1-self.data['label'])
        after=long_history(changed,np.array([2]),self.clicks,11)
        for a,b in zip(before,after):np.testing.assert_array_equal(a,b)

    def test_other_context_omits_exact_target_and_keeps_user_boundaries(self):
        records=np.array([(0,20,1),(0,30,2),(0,40,3),(1,50,4)],dtype=self.clicks.dtype)
        h,n,target=other_click_context(records,limit=2)
        self.assertEqual(target.tolist(),[1,2,3])
        self.assertEqual(n.tolist(),[2,2,2])
        for context,t in zip(h,target):self.assertNotIn(t,context)
        self.assertTrue(np.all(h<=3))

    def test_ensemble_has_shared_items_and_only_scalar_user_bias(self):
        model=SharedHistoryEnsemble(3,11,seeds=3,dim=4)
        self.assertEqual(model.user_bias.weight.shape,(9,1))
        self.assertEqual(model.item.weight.shape,(33,4))
        self.assertFalse(hasattr(model,'history'))
        self.assertFalse(hasattr(model,'user'))

    def test_finite_empty_history_and_candidate_permutation(self):
        model=SharedHistoryEnsemble(3,11,seeds=3,dim=4)
        uid=torch.tensor([[0,1]]).expand(3,-1)
        items=torch.arange(1,11)[None,None].expand(3,2,-1)
        hist=torch.zeros(3,2,10,dtype=torch.long)
        hist[:,0,:2]=torch.tensor([1,2])
        lengths=torch.tensor([[2.,0.]]).expand(3,-1)
        parts,reg=model(uid,items,hist,lengths,regularize=True)
        self.assertTrue(torch.isfinite(parts).all())
        torch.testing.assert_close(parts[:,1,:,3],torch.zeros(3,10))
        perm=torch.tensor([2,1,5,7,3,6,0,8,4,9])
        torch.testing.assert_close(parts[:,:,perm],model(uid,items[:,:,perm],hist,lengths))
        before=parts.detach().clone()
        with torch.no_grad():model.item.weight[:11].add_(1)
        after=model(uid,items,hist,lengths)
        torch.testing.assert_close(before[1:],after[1:])
        self.assertGreater(float((before[0]-after[0]).detach().abs().sum()),0)
        # Recompute after the parameter mutation to keep autograd versioning valid.
        p,r=model(uid,items,hist,lengths,regularize=True)
        (p[...,0].sum()+r.sum()).backward()
        self.assertGreater(float(model.item.weight.grad.abs().sum()),0)
        for seed in range(3):torch.testing.assert_close(model.item.weight.grad[seed*11],torch.zeros(4))

    def test_vectorized_seeds_match_single_model_values_and_gradients(self):
        multi=SharedHistoryEnsemble(3,11,seeds=3,dim=4)
        single=SharedHistoryEnsemble(3,11,seeds=1,dim=4)
        uid=torch.tensor([[0,1]]).expand(3,-1)
        items=torch.arange(1,11)[None,None].expand(3,2,-1)
        hist=torch.tensor([[[1,2,0],[3,0,0]]]).expand(3,-1,-1)
        lengths=torch.tensor([[2.,1.]]).expand(3,-1)
        a,ar=multi(uid,items,hist,lengths,regularize=True)
        b,br=single(uid[:1],items[:1],hist[:1],lengths[:1],regularize=True)
        torch.testing.assert_close(a[:1],b)
        (a[...,0].square().mean((1,2))+.05*ar).sum().backward()
        (b[...,0].square().mean((1,2))+.05*br).sum().backward()
        torch.testing.assert_close(multi.item.weight.grad[:11],single.item.weight.grad)

    def test_interrupted_fit_resumes_exactly_on_cpu(self):
        args=argparse.Namespace(device='cpu',seeds=2,steps=12,batch_events=2,checkpoint_every=4)
        sizes={'all_users':3,'all_items':10}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(upgrade,'OUT',root/'continuous'):
                expected,_=upgrade.fit_ensemble(self.data,2,sizes,args)
            save=upgrade.atomic_checkpoint
            def interrupt(path,value):
                save(path,value)
                if value.get('step')==8:raise InterruptedError('Simulated shutdown after a saved step')
            with patch.object(upgrade,'OUT',root/'resumed'):
                with patch.object(upgrade,'atomic_checkpoint',interrupt):
                    with self.assertRaises(InterruptedError):upgrade.fit_ensemble(self.data,2,sizes,args)
                actual,_=upgrade.fit_ensemble(self.data,2,sizes,args)
            for key,value in expected.state_dict().items():
                torch.testing.assert_close(value,actual.state_dict()[key],rtol=0,atol=0)

    def test_default_recipe_keeps_baseline_identity(self):
        # Saved baseline checkpoints are keyed by this dict; new flags must not change it.
        args=argparse.Namespace(seeds=10,steps=5000,batch_events=820)
        self.assertEqual(upgrade.recipe(args),dict(version=1,dim=32,l2=.05,lr=.003,auxiliary_weight=.1,
            negatives=9,history_limit=100,init_std=.05,seeds=10,steps=5000,batch_events=820,
            event_sampling='independent uniform with replacement per seed',
            historical_inputs='fitting histories and labels only; own query history at inference'))

    def test_tree_only_settings_do_not_change_the_mf_identity(self):
        # Tree-only experiments must be able to reuse a parent run's MF models and parts.
        base=argparse.Namespace(seeds=30,steps=3000,batch_events=820,init_std=.005)
        subset=argparse.Namespace(seeds=30,steps=3000,batch_events=820,init_std=.005,feature_subset='no-spectral')
        with patch.object(upgrade.common,'cache_fingerprint',lambda:'x'):
            self.assertEqual(upgrade.mf_identity(base),upgrade.mf_identity(subset))
            self.assertEqual(upgrade.run_identity(base),upgrade.mf_identity(base))
            self.assertEqual(upgrade.run_identity(subset)['feature_subset'],'no-spectral')

    def test_a_directory_with_another_recipe_is_refused_before_it_changes(self):
        args=argparse.Namespace(seeds=10,steps=5000,batch_events=820)
        with tempfile.TemporaryDirectory() as directory,patch.object(upgrade.common,'cache_fingerprint',lambda:'x'):
            with patch.object(upgrade,'OUT',Path(directory)):
                upgrade.write_recipe(args)
                saved=(Path(directory)/'recipe.json').read_text()
                upgrade.write_recipe(args)
                with self.assertRaisesRegex(ValueError,'different recipe'):
                    upgrade.write_recipe(argparse.Namespace(seeds=10,steps=3000,batch_events=820))
                self.assertEqual((Path(directory)/'recipe.json').read_text(),saved)

    def test_every_feature_subset_excludes_the_old_learned_mf_features(self):
        names=['item_ctr','neural_score','learned_dot','candidate_vector_norm','history_vector_norm',
               'spectral_rank','cf_rank','starter_logit','item_interacting_users_log']
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)/'artifacts/strategy2/features'
            folder.mkdir(parents=True)
            (folder/'train.json').write_text(json.dumps({'feature_names':names,'identity':{'data_fingerprint':'x'}}))
            with patch.object(upgrade,'ROOT',Path(directory)),patch.object(upgrade.common,'cache_fingerprint',lambda:'x'):
                expected={'baseline':['item_ctr','spectral_rank','cf_rank','starter_logit','item_interacting_users_log'],
                          'no-extended':['item_ctr','spectral_rank','cf_rank','starter_logit','item_interacting_users_log'],
                          'no-spectral':['item_ctr','cf_rank','starter_logit','item_interacting_users_log'],
                          'reference':['item_ctr']}
                for subset,kept in expected.items():
                    self.assertEqual(upgrade.base_columns(subset)[1],kept)

    def test_scored_query_clicks_are_real_pre_cutoff_history_clicks(self):
        day,cutoff=2,upgrade.START+2*upgrade.DAY
        data={'user':np.array([5,5,7]),'time':np.array([cutoff-10,cutoff+10,cutoff+20]),
              'hist':np.array([[3]+[-1]*19,[3,4,8]+[-1]*17,[6,9]+[-1]*18]),
              'hist_time':np.array([[cutoff-50]+[0]*19,[cutoff-50,cutoff-40,cutoff+5]+[0]*17,[cutoff-30,cutoff+1]+[0]*18])}
        clicks=upgrade.scored_query_clicks(data,day)
        # Item 8 and item 9 are after the cutoff; the first event belongs to the previous day.
        self.assertEqual(sorted(clicks[['user','item']].tolist()),[(5,4),(5,5),(7,7)])
        self.assertTrue((clicks['time']<cutoff).all())

    def stale_clicks(self):
        hour=3600
        rows=[(1,t*hour,i) for t,i in [(10,11),(20,12),(30,13),(31,14),(32,15)]]+[(2,5*hour,16)]
        return np.array(rows,dtype=[('user','i4'),('time','i8'),('item','i4')])

    def test_stale_context_hours_really_removes_clicks_near_the_target(self):
        # stale_context_hours: the same clicks must not stay in the context array.
        clicks=self.stale_clicks()
        plain,plain_n,plain_targets=other_click_context(clicks,limit=10)
        gapped,gapped_n,gapped_targets=other_click_context(clicks,limit=10,gap=5*3600)
        row=lambda m,target:m[plain_targets.tolist().index(target) if m is plain else gapped_targets.tolist().index(target)]
        # Target item 14 at hour 31: items 13 and 15 are within 5 hours, 11 and 12 are not.
        self.assertEqual(sorted(i for i in row(plain,14).tolist() if i),[11,12,13,15])
        self.assertEqual(sorted(i for i in row(gapped,14).tolist() if i),[11,12])
        index=gapped_targets.tolist().index(14)
        self.assertEqual(int(gapped_n[index]),2)
        # Lengths always equal the number of clicks kept, so the average is not rescaled.
        self.assertEqual(((gapped>0).sum(axis=1)==gapped_n).all(),True)
        self.assertEqual(((plain>0).sum(axis=1)==plain_n).all(),True)

    def test_stale_context_keeps_later_clicks_outside_the_gap(self):
        gapped,_,targets=other_click_context(self.stale_clicks(),limit=10,gap=5*3600)
        # Target item 11 at hour 10: later clicks at hours 20-32 are outside the gap and kept.
        self.assertEqual(sorted(i for i in gapped[targets.tolist().index(11)].tolist() if i),[12,13,14,15])

    def test_stale_context_drops_targets_left_with_no_context(self):
        _,_,targets=other_click_context(self.stale_clicks(),limit=10,gap=100*3600)
        self.assertEqual(targets.tolist(),[])

    def test_no_gap_keeps_the_original_context_exactly(self):
        clicks=self.stale_clicks()
        plain=other_click_context(clicks,limit=3)
        self.assertEqual(plain[0].tolist(),[[15,14,13],[15,14,13],[15,14,12],[15,13,12],[14,13,12]])
        self.assertEqual(plain[1].tolist(),[3,3,3,3,3])
        self.assertEqual(plain[2].tolist(),[11,12,13,14,15])

    def test_stale_history_hours_really_shortens_training_histories(self):
        # stale_history_hours: both the click log and the event's own history respect the gap.
        h,n=long_history(self.data,np.array([1]),self.clicks,11,limit=10)
        gapped,gapped_n=long_history(self.data,np.array([1]),self.clicks,11,limit=10,gap=120)
        self.assertEqual(n.tolist(),[3])
        self.assertEqual(sorted(i for i in h[0].tolist() if i),[1,1,2])
        # The event is at time 200, so only clicks before 80 remain: items 0 and 1 at 20 and 30.
        self.assertEqual(gapped_n.tolist(),[2])
        self.assertEqual(sorted(i for i in gapped[0].tolist() if i),[1,2])

    def test_stale_settings_enter_the_recipe_only_when_used(self):
        base=argparse.Namespace(seeds=10,steps=3000,batch_events=820)
        self.assertNotIn('stale_context_hours',upgrade.recipe(base))
        used=argparse.Namespace(seeds=10,steps=3000,batch_events=820,stale_context_hours=24.,stale_history_hours=0.)
        self.assertEqual(upgrade.recipe(used)['stale_context_hours'],24.)
        self.assertNotIn('stale_history_hours',upgrade.recipe(used))

    def test_stale_history_hours_entered_when_used(self):
        # stale_history_hours must appear in recipe when non-zero (different from stale-context which is auxiliary).
        base=argparse.Namespace(seeds=10,steps=3000,batch_events=820)
        self.assertNotIn('stale_history_hours',upgrade.recipe(base))
        used=argparse.Namespace(seeds=10,steps=3000,batch_events=820,stale_history_hours=12.)
        self.assertIn('stale_history_hours',upgrade.recipe(used))
        self.assertEqual(upgrade.recipe(used)['stale_history_hours'],12.)

    def test_per_seed_steps_stop_each_seed_exactly_where_uniform_training_would(self):
        sizes={'all_users':3,'all_items':10}
        def trained(**kwargs):
            args=argparse.Namespace(device='cpu',seeds=2,batch_events=2,checkpoint_every=2,**kwargs)
            with tempfile.TemporaryDirectory() as directory,patch.object(upgrade,'OUT',Path(directory)):
                return upgrade.fit_ensemble(self.data,2,sizes,args)[0]
        mixed=trained(steps=4,steps_per_seed=[2,4])
        short,full=trained(steps=2),trained(steps=4)
        for got,want in zip(mixed.seed_state(0),short.seed_state(0)):
            torch.testing.assert_close(got,want,rtol=0,atol=0)
        for got,want in zip(mixed.seed_state(1),full.seed_state(1)):
            torch.testing.assert_close(got,want,rtol=0,atol=0)

    def test_interrupted_per_seed_fit_resumes_exactly_on_cpu(self):
        sizes={'all_users':3,'all_items':10}
        args=argparse.Namespace(device='cpu',seeds=2,steps=6,batch_events=2,checkpoint_every=2,steps_per_seed=[2,6])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(upgrade,'OUT',root/'continuous'):
                expected,_=upgrade.fit_ensemble(self.data,2,sizes,args)
            save=upgrade.atomic_checkpoint
            def interrupt(path,value):
                save(path,value)
                if value.get('step')==4:raise InterruptedError('Simulated shutdown after a saved step')
            with patch.object(upgrade,'OUT',root/'resumed'):
                with patch.object(upgrade,'atomic_checkpoint',interrupt):
                    with self.assertRaises(InterruptedError):upgrade.fit_ensemble(self.data,2,sizes,args)
                actual,_=upgrade.fit_ensemble(self.data,2,sizes,args)
            for key,value in expected.state_dict().items():
                torch.testing.assert_close(value,actual.state_dict()[key],rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
