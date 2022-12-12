import pytest 
import numpy as np

class TestCoverage:

  def test_init(self):
    f = open("results.txt", "r")
    temp= f.readline(1)
    assert(temp == '9')

