"""Checks for deduplication, temporal boundaries, masking, and independent seeds."""
import unittest
import argparse
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import torch
import history_upgrade as upgrade
from history_upgrade import unique_clicks,long_history,other_click_context,SharedHistoryEnsemble


class HistoryUpgradeTests(unittest.TestCase):
    def setUp(self):
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


if __name__=='__main__':
    unittest.main()
