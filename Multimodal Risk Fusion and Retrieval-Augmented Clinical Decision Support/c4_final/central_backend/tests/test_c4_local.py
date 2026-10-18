"""Regression tests against actual DCAR holdout golden predictions."""
import json
import math
from pathlib import Path
import pytest
import modality_clients as mc
from components.c4.inference import load_model, model_available, predict_demographics

BASE={'age':21,'gender':'female','edu':"bachelor's degree",
      'smoke':'never smokes','drink':'never drinks'}

@pytest.mark.parametrize('age,gender,edu,smoke,drink,expected',[
  (18,'male',"master's degree",'never smokes','never drinks',.0068296923825597),
  (21,'female',"bachelor's degree",'never smokes','never drinks',.0182847334986927),
  (22,'female',"bachelor's degree",'current smoker (cumulative smoking >10 packs)',
   'current regular drinker (more than once a week)',.1966078407311266),
])
def test_golden_predictions(age,gender,edu,smoke,drink,expected):
    result=predict_demographics({'age':age,'gender':gender,'edu':edu,'smoke':smoke,'drink':drink})
    assert math.isclose(result['demographic_probability'],expected,abs_tol=1e-13)
    assert result['target']=='GAD-7 >= 10'
    assert result['score_semantics']=='calibrated_probability_gad7_ge10'

@pytest.mark.parametrize('invalid',[
  {'age':None},{'age':float('nan')},{'age':float('inf')},
  {'age':17},{'age':28},{'age':True},{'gender':'unknown'},
  {'edu':None},{'smoke':'unknown'},{'drink':None}
])
def test_unsupported_profiles_not_fused(invalid):
    result=mc.call_c4('synthetic-subject',{**BASE,**invalid})
    assert result.raw_score is None
    assert result.status=='poor_signal'
    assert result.coverage==0

def test_local_adapter_never_uses_external_http(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('C4 attempted external network call')
    monkeypatch.setattr(mc.httpx,'Client',forbidden)
    monkeypatch.setattr(mc,'C4_BASE','https://not-used.invalid')
    assert mc.CONFIGURED['c4_demographic']()
    r=mc.call_c4('synthetic-subject',BASE)
    assert r.status=='ok'
    assert math.isclose(r.raw_score,.0182847334986927,abs_tol=1e-13)
    assert r.detail['demographic_fusion_score'] != r.raw_score
    assert r.confidence==0.5
    assert r.model_version==load_model()['model_version']

def test_reference_provenance_and_ties():
    from fusion_client import _fuse_inprocess
    root=Path(__file__).resolve().parents[2]
    ref=json.loads((root/'fusion_service/reference/c4_demographic.json').read_text())
    assert ref['n']==4854 and sum(ref['counts'])==4854
    assert len(ref['scores'])==len(ref['counts'])==258
    assert 'PLACEHOLDER' not in ref['source']
    assert ref['score_type']=='calibrated_probability_gad7_ge10'
    assert ref['model_version']==load_model()['model_version']
    import sys
    sys.path.insert(0,str(root/'fusion_service'))
    from harmonise import Harmoniser
    h=Harmoniser(root/'fusion_service/reference')
    assert len(h.refs['c4_demographic'])==4854
    assert abs(h.refs['c4_demographic'].mean()-ref['mean'])<1e-12
    p=predict_demographics(BASE)['demographic_probability']
    q=h.harmonise('c4_demographic',p)
    assert 0<q.value<1

def test_training_labels_not_accepted_as_features():
    r1=mc.call_c4('synthetic-subject',BASE)
    r2=mc.call_c4('synthetic-subject',{**BASE,'gad7_items':[3]*7,'gad7_total':21})
    assert r1.raw_score==r2.raw_score
    assert model_available()
