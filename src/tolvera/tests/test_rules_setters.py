'''
TODO: segfaults, how do Taichi run their tests?
'''

import pytest
import taichi as ti
import numpy as np
from tolvera import Tolvera
from tolvera.utils import _show
from tolvera import Rules

@pytest.fixture
def tolvera_setup():
    return Tolvera(n=1024, species=2, evaporate=0.99, headless=True)

# Test cases
def test_set_kwargs(tolvera_setup):
    tv = tolvera_setup
    tv.flock.rules.set_kwargs((0, 0), radius=100.0)
    # Add an assertion here to verify that the kwargs were set correctly

def test_set_args(tolvera_setup):
    tv = tolvera_setup
    tv.flock.rules.set_args((0, 0), 1.0, 1.0, 0.1, 100.0)
    # Add an assertion here to verify that the args were set correctly

def test_set_list(tolvera_setup):
    tv = tolvera_setup
    rule_list = [0.1, 0.2, 0.7, 50.0]
    tv.flock.rules.set_list((0, 0), rule_list)
    # Add an assertion here to verify that the list was set correctly

def test_set_all_ndarray(tolvera_setup):
    tv = tolvera_setup
    rules_all_ndarray = np.zeros((tv.o.species, tv.o.species, len(tv.flock.rules.dict)), dtype=np.float32)
    tv.flock.rules.set_all_ndarray(rules_all_ndarray)
    # Add an assertion here to verify that the ndarray was set correctly

def test_set_all_list(tolvera_setup):
    tv = tolvera_setup
    rules_all_list = [0.1, 0.2, 0.7, 50.0, 0.1, 0.2, 0.7, 200.0, 0.1, 0.2, 0.7, 50.0, 0.1, 0.2, 0.7, 200.0]
    tv.flock.rules.set_all_list(rules_all_list)
    # Add an assertion here to verify that the list was set correctly

def test_set_species_ndarray(tolvera_setup):
    tv = tolvera_setup
    rules_species_ndarray = np.zeros((tv.o.species, len(tv.flock.rules.dict)), dtype=np.float32)
    tv.flock.rules.set_species_ndarray(0, rules_species_ndarray)
    # Add an assertion here to verify that the ndarray was set correctly

def test_set_species_list(tolvera_setup):
    tv = tolvera_setup
    rules_species_list = [0.1, 0.2, 0.7, 50.0, 0.1, 0.2, 0.7, 200.0]
    tv.flock.rules.set_species_list(0, rules_species_list)
    # Add an assertion here to verify that the list was set correctly
