import pytest 
import numpy as np
from Autodiff43.logic.forward_mode import FMExpression
from Autodiff43.logic.reverse_mode import RMExpression

"""
Expression

__mul__
__add__
__sub__
reflective operations
trig functions
exponentinal

Test Suite:

- operation tests
- init test: supports correct data formats
- 
"""

class TestExpression:
    # Tests for Forward Mode, FMExpression

    def test_init_FM(self):
        expression_init_test = FMExpression(1,"x")
        assert(expression_init_test.value == 1)
        assert(expression_init_test.grad == {"x": [1]})

    def test_addition_FM(self):
        #test scalar addition
        e1 = FMExpression(1, "x")
        e2= e1 + (-1)
        assert(e2.value == 0)
        assert(FMExpression.grad(e2,"x") == 1)
        
        #test for type errors
        s= "abc"
        with pytest.raises(TypeError):
            e1 + s

        #test expression + expression
        e1 = FMExpression(2, "x")
        e2 = FMExpression(1, "y")
        e3 = e1 + e2
        assert(e3.value == 3)
        assert(FMExpression.grad(e3,"x") == 1)

        #try if the keys are already in grad
        e4 = e3 + e3
        assert(e4.value == 6)
        assert(FMExpression.grad(e4,"x") == 2)

        #test radd
        e1 = FMExpression(1, "x")
        e2 = 1 + e1
        assert(e2.value == 2)
        assert(FMExpression.grad(e2,"x") == 1)

    def test_subtraction_FM(self):
        #test scalar subtraction
        e1 = FMExpression(2, "x")
        e2 = e1 - 1
        assert(e2.value == 1)
        assert(FMExpression.grad(e2,"x") == 1)
        
        #test for type errors
        s = "abc"
        with pytest.raises(TypeError):
            e1 - s

        #test expression - expression
        e1 = FMExpression(2, "x")
        e2 = FMExpression(1, "y")
        e3 = e1 - e2
        assert(e3.value == 1)
        assert(FMExpression.grad(e3,"y") == -1)

        #try if the keys are already in grad
        e4 = e3 - e3
        assert(e4.value == 0)
        assert(FMExpression.grad(e4,"x") == 0)

        #test scalar subtraction reverse
        e1 = FMExpression(2, "x")
        e2 = 1 - e1
        assert(e2.value == -1)
        assert(FMExpression.grad(e2,"x") == -1)

        s= "abc"
        with pytest.raises(TypeError):
            s - e1

    def test_mul_FM(self):
        #test neg
        e0 = FMExpression(2, "x")
        e0 = -e0
        assert(e0.value == -2)
        assert(FMExpression.grad(e0,"x") == -1)

        #test with scalar
        e1 = FMExpression(2, "x")
        e2= e1 * 3
        assert(e2.value == 6)
        assert(FMExpression.grad(e2,"x") == 3)

        #test between expressions
        e3 = FMExpression(3, "y")
        e4= e3 * e1
        assert(e4.value == 6)
        assert(FMExpression.grad(e4,"x") == 3)

        #type error test
        s = "abc"
        with pytest.raises(TypeError):
            e1 * s

        #reverse test
        e1 = FMExpression(2, "x")
        e2 = 3 * e1
        assert(e2.value == 6)
        assert(FMExpression.grad(e2,"x") == 3)

    def test_div_FM(self):
        #with scalar
        e1 = FMExpression(6,"x")
        e2= e1 / 3
        assert(e2.value == 2)
        assert(FMExpression.grad(e2,"x") == 1/3)

        #between expressions
        e3 = FMExpression(3, "y")
        e4 = e1 / e3
        assert(e4.value == 2)
        assert(FMExpression.grad(e4,"x") == 1/3)

        #type error test
        s= "abc"
        with pytest.raises(TypeError):
            e1 / s

        #divide by 0
        e5 = FMExpression(0, "x")
        with pytest.raises(ZeroDivisionError):
            e1 / 0
        with pytest.raises(ZeroDivisionError):
            e1 / e5
        
        #reverse test
        e1 = FMExpression(6,"x")
        e2 = 3 / e1
        assert(e2.value == 1/2)
        assert(FMExpression.grad(e2,"x") == -1/12)

        #reverse test divid by 0
        e1 = FMExpression(0,"x")
        with pytest.raises(ZeroDivisionError):
            3 / e1

        #type error test for reverse
        s = "abc"
        with pytest.raises(TypeError):
            s / e1
    
    def test_pow_FM(self):
        e1 = FMExpression(2,"x")
        e2 = FMExpression(1,"y")

        e3 = e1 ** e2
        e4 = e1 ** 2
        assert(e3.value == 2)
        assert(FMExpression.grad(e3,"x") == 1)

        assert(e4.value == 4)
        assert(FMExpression.grad(e4,"x") == 4)

        s= "abc"
        with pytest.raises(TypeError):
            e1 ** s
        
        #test sqrt
        e5 = FMExpression(4,"x")
        e5 = FMExpression.sqrt(e5)
        assert(e5.value == 2)
        assert(FMExpression.grad(e5,"x") == 1/4)

    def test_sin_FM(self):
        e0 = FMExpression(1,"x")
        e1 = FMExpression.sin(e0)

        assert e1.value == np.sin(1)
        assert FMExpression.grad(e1,"x") == np.cos(1)

        #test with exponent
        e2 = e0 ** 2
        e3 = FMExpression.sin(e2)
        assert e3.value == np.sin(1)
        assert FMExpression.grad(e3,"x") == np.cos(1) * 2

    def test_cos_FM(self):
        e0 = FMExpression(1,"x")
        e1 = FMExpression.cos(e0)

        assert e1.value == np.cos(1)
        assert FMExpression.grad(e1,"x") == -1 * np.sin(1)

        #test with exponent
        e2 = e0 ** 2
        e3 = FMExpression.cos(e2)
        assert e3.value == np.cos(1)
        assert FMExpression.grad(e3,"x") == -1 * np.sin(1) * 2
    
    def test_tan_FM(self):
        e0 = FMExpression(1,"x")
        e1 = FMExpression.tan(e0)

        assert e1.value == np.tan(1)
        assert FMExpression.grad(e1,"x") == 1 / (np.cos(1)* np.cos(1))

        #test with exponent
        e2 = e0 ** 2
        e3 = FMExpression.tan(e2)
        assert e3.value == np.tan(1)
        assert FMExpression.grad(e3,"x") == 2 / (np.cos(1)* np.cos(1))
    
    def test_arcsin_FM(self):
        e0 = FMExpression(1 / 2,"x")
        e1 = FMExpression.arcsin(e0)

        assert e1.value == np.arcsin(1 / 2)
        assert FMExpression.grad(e1,"x") == 1 / np.sqrt(1 - (1 / 2) ** 2)
    
    def test_arccos_FM(self):
        e0 = FMExpression(1/2,"x")
        e1 = FMExpression.arccos(e0)

        assert e1.value == np.arccos(1 / 2)
        assert FMExpression.grad(e1,"x") == -1 / np.sqrt(1 - (1 / 2) ** 2)
    
    def test_arctan_FM(self):
        e0 = FMExpression(1/2,"x")
        e1 = FMExpression.arctan(e0)

        assert e1.value == np.arctan(1 / 2)
        assert FMExpression.grad(e1,"x") == 1 / (1 + (1 / 2) ** 2)

    def test_sinh_FM(self):
        e0 = FMExpression(3 / 2,"x")
        e1 = FMExpression.sinh(e0)

        assert e1.value == np.sinh(3 / 2)
        assert FMExpression.grad(e1,"x") == np.cosh(3 / 2)
    
    def test_cosh_FM(self):
        e0 = FMExpression(3 / 2,"x")
        e1 = FMExpression.cosh(e0)

        assert e1.value == np.cosh(3 / 2)
        assert FMExpression.grad(e1,"x") == np.sinh(3 / 2)
    
    def test_tanh_FM(self):
        e0 = FMExpression(3 / 2,"x")
        e1 = FMExpression.tanh(e0)

        assert e1.value == np.tanh(3 / 2)
        assert FMExpression.grad(e1,"x") == 1 / (np.cosh(3 / 2) ** 2)

    def test_log_FM(self):
        e0 = FMExpression(2,"x")
        e1 = FMExpression.log(e0, 10)

        assert e1.value == np.log(2) / np.log(10)
        assert FMExpression.grad(e1,"x") == 1 / (2 * np.log(10))

    def test_sigmoid_FM(self):
        e0 = FMExpression(2,"x")
        e1 = FMExpression.sigmoid(e0)

        assert e1.value == 1 / (1 + np.exp(-2))
        assert FMExpression.grad(e1,"x") == np.exp(2) / ((np.exp(2) + 1) ** 2)
    
    def test_exp_FM(self):
        e0 = FMExpression(2,"x")
        e1 = FMExpression.exp(e0)

        assert e1.value == np.exp(2)
        assert FMExpression.grad(e1,"x") == np.exp(2)

        # exp where the base is not e
        e0 = FMExpression(2,"x")
        e1 = FMExpression.exp(e0, 10)

        assert e1.value == 10 ** 2
        assert FMExpression.grad(e1,"x") == (10 ** 2) * np.log(10)
    
    def test_vec_FM(self):
        e0 = FMExpression(2,"x")
        e1 = FMExpression(3,"y")
        e2 = FMExpression.vec(e0 * e1, e0 + e1)
        assert len(e2.value)==2
        assert e2.value[0] == 6
        assert e2.value[1] == 5
        assert FMExpression.grad(e2, 0, "x") == 3
        assert FMExpression.grad(e2, 1, "x") == 1

        e3 = FMExpression.vec(e0 * 2,e0-1)
        assert len(e2.value) == 2
        assert e3.value[0] == 4
        assert e3.value[1] == 1
        assert FMExpression.grad(e3, 0, "x") == 2
        assert FMExpression.grad(e3, 1, "x") == 1

    # Tests for Reverse Mode, RMExpression

    def test_init_RM(self):
        expression_init_test = RMExpression([1,2,3], "x", [1, 2, 3])
        assert(expression_init_test.node_edges == [1,2,3])
        expression_init_test = RMExpression(1, "x")
        assert(RMExpression.grad(expression_init_test, "x") == 1)

    def test_addition_RM(self):
        #test scalar addition
        e1 = RMExpression(1, "x")
        e2 = e1 + (-1)
        assert(e2.value == 0)
        assert(RMExpression.grad(e2,"x") == 1)
        
        #test for type errors
        s = "abc"
        with pytest.raises(TypeError):
            e1 + s

        #test expression + expression
        e1 = RMExpression(2, "x")
        e2 = RMExpression(1, "y")
        e3 = e1 + e2
        assert(e3.value == 3)
        assert(RMExpression.grad(e3,"x") == 1)

        #try if the keys are already in grad
        e4 = e3 + e3
        assert(e4.value == 6)
        assert(RMExpression.grad(e4,"x") == 2)

        #test radd
        e1 = RMExpression(1, "x")
        e2 = 1 + e1
        assert(e2.value == 2)
        assert(RMExpression.grad(e2,"x") == 1)
    
    def test_subtraction_RM(self):
        #test scalar subtraction
        e1 = RMExpression(2, "x")
        e2 = e1 - 1
        assert(e2.value == 1)
        assert(RMExpression.grad(e2,"x") == 1)
        
        #test for type errors
        s = "abc"
        with pytest.raises(TypeError):
            e1 - s

        #test expression - expression
        e1 = RMExpression(2, "x")
        e2 = RMExpression(1, "y")
        e3 = e1 - e2
        assert(e3.value == 1)
        assert(RMExpression.grad(e3,"y") == -1)

        #try if the keys are already in grad
        e4 = e3 - e3
        assert(e4.value == 0)
        assert(RMExpression.grad(e4,"x") == 0)

        #test scalar subtraction reverse
        e1 = RMExpression(2, "x")
        e2 = 1 - e1
        assert(e2.value == -1)
        assert(RMExpression.grad(e2,"x") == -1)
        # np.testing.assert_array_equal(e2.value, np.array([1]))

        s = "abc"
        with pytest.raises(TypeError):
            s - e1
    
    def test_mul_RM(self):
        #test neg
        e0 = RMExpression(2, "x")
        e0 = -e0
        assert(e0.value == -2)

        #test with scalar
        e1 = RMExpression(2, "x")
        e2= e1 * 3
        assert(e2.value == 6)
        assert(RMExpression.grad(e2,"x") == 3)

        #test between expressions
        e3 = RMExpression(3, "y")
        e4= e3 * e1
        assert(e4.value == 6)
        assert(RMExpression.grad(e4,"x") == 3)

        #type error test
        s = "abc"
        with pytest.raises(TypeError):
            e1 * s

        #reverse test
        e1 = RMExpression(2, "x")
        e2 = 3 * e1
        assert(e2.value == 6)
        assert(RMExpression.grad(e2,"x") == 3)

    def test_div_RM(self):
        #with scalar
        e1 = RMExpression(6, "x")
        e2= e1 / 3
        assert(e2.value == 2)
        assert(RMExpression.grad(e2,"x") == 1/3)

        #between expressions
        e3 = RMExpression(3, "y")
        e4 = e1 / e3
        assert(e4.value == 2)
        assert(RMExpression.grad(e4,"x") == 1/3)

        #type error test
        s = "abc"
        with pytest.raises(TypeError):
            e1 / s

        #divide by 0
        e5 = RMExpression(0, "x")
        with pytest.raises(ZeroDivisionError):
            e1 / 0
        with pytest.raises(ZeroDivisionError):
            e1 / e5

        #reverse test
        e1 = RMExpression(6,"x")
        e2 = 3 / e1
        assert(e2.value == 1/2)
        assert(RMExpression.grad(e2,"x") == -1/12)

        #type error test for reverse
        s = "abc"
        with pytest.raises(TypeError):
            s / e1
    
    def test_pow_RM(self):
        e1 = RMExpression(2, "x")
        e2 = RMExpression(1, "y")

        e3 = e1 ** e2
        e4 = e1 ** 2
        assert(e3.value == 2)
        assert(RMExpression.grad(e3,"x") == 1)

        assert(e4.value == 4)
        assert(RMExpression.grad(e4,"x") == 4)

        s = "abc"
        with pytest.raises(TypeError):
            e1 ** s
        
        #test sqrt
        e5 = RMExpression(4, "x")
        e5 = RMExpression.sqrt(e5)
        assert(e5.value == 2)
        assert(RMExpression.grad(e5,"x") == 1/4)

    def test_sin_RM(self):
        e0 = RMExpression(1, "x")
        e1 = RMExpression.sin(e0)

        assert e1.value == np.sin(1)
        assert RMExpression.grad(e1,"x") == np.cos(1)

        #test with exponent
        e2 = e0 ** 2
        e3 = RMExpression.sin(e2)
        assert e3.value == np.sin(1)
        assert RMExpression.grad(e3,"x") == np.cos(1) * 2

    def test_cos_RM(self):
        e0 = RMExpression(1, "x")
        e1 = RMExpression.cos(e0)

        assert e1.value == np.cos(1)
        assert RMExpression.grad(e1,"x") == -1 * np.sin(1)

        #test with exponent
        e2 = e0 ** 2
        e3 = RMExpression.cos(e2)
        assert e3.value == np.cos(1)
        assert RMExpression.grad(e3,"x") == -1 * np.sin(1) * 2

    def test_tan_RM(self):
        e0 = RMExpression(1, "x")
        e1 = RMExpression.tan(e0)

        assert e1.value == np.tan(1)
        assert RMExpression.grad(e1,"x") == 1 / (np.cos(1)* np.cos(1))

        #test with exponent
        e2 = e0 ** 2
        e3 = RMExpression.tan(e2)
        assert e3.value == np.tan(1)
        assert RMExpression.grad(e3,"x") == 2 / (np.cos(1)* np.cos(1))
    
    def test_arcsin_RM(self):
        e0 = RMExpression(1 / 2, "x")
        e1 = RMExpression.arcsin(e0)

        assert e1.value == np.arcsin(1 / 2)
        assert RMExpression.grad(e1, "x") == 1 / np.sqrt(1 - (1 / 2) ** 2)
    
    def test_arccos_RM(self):
        e0 = RMExpression(1 / 2, "x")
        e1 = RMExpression.arccos(e0)

        assert e1.value == np.arccos(1 / 2)
        assert RMExpression.grad(e1,"x") == -1 / np.sqrt(1 - (1 / 2) ** 2)
    
    def test_arctan_RM(self):
        e0 = RMExpression(1 / 2, "x")
        e1 = RMExpression.arctan(e0)

        assert e1.value == np.arctan(1 / 2)
        assert RMExpression.grad(e1,"x") == 1 / (1 + (1 / 2) ** 2)

    def test_sinh_RM(self):
        e0 = RMExpression(3 / 2, "x")
        e1 = RMExpression.sinh(e0)

        assert e1.value == np.sinh(3 / 2)
        assert RMExpression.grad(e1,"x") == np.cosh(3 / 2)
    
    def test_cosh_RM(self):
        e0 = RMExpression(3 / 2, "x")
        e1 = RMExpression.cosh(e0)

        assert e1.value == np.cosh(3 / 2)
        assert RMExpression.grad(e1,"x") == np.sinh(3 / 2)
    
    def test_tanh_RM(self):
        e0 = RMExpression(3 / 2, "x")
        e1 = RMExpression.tanh(e0)

        assert e1.value == np.tanh(3 / 2)
        assert RMExpression.grad(e1,"x") == 1 / (np.cosh(3 / 2) ** 2)

    def test_log_RM(self):
        e0 = RMExpression(2,"x")
        e1 = RMExpression.log(e0, 10)

        assert e1.value == np.log(2) / np.log(10)
        assert RMExpression.grad(e1,"x") == 1 / (2 * np.log(10))

    def test_sigmoid_RM(self):
        e0 = RMExpression(2,"x")
        e1 = RMExpression.sigmoid(e0)

        assert e1.value == 1 / (1 + np.exp(-2))
        assert RMExpression.grad(e1,"x") == np.exp(2) / ((np.exp(2) + 1) ** 2)

    def test_exp_RM(self):
        e0 = RMExpression(2,"x")
        e1 = RMExpression.exp(e0)

        assert e1.value == np.exp(2)
        assert RMExpression.grad(e1,"x") == np.exp(2)

        # exp where the base is not e
        e0 = RMExpression(2,"x")
        e1 = RMExpression.exp(e0, 10)

        assert e1.value == 10 ** 2
        assert RMExpression.grad(e1,"x") == (10 ** 2) * np.log(10)

    def test_vec_RM(self):
        e0 = RMExpression(2,"x")
        e1 = RMExpression(3,"y")
        e2 = RMExpression.vec(e0 * e1, e0 + e1)
        assert len(e2)==2
        assert e2[0].value == 6
        assert e2[1].value == 5
        assert RMExpression.grad(e2, 0, "x") == 3
        assert RMExpression.grad(e2, 1, "x") == 1

        e3 = RMExpression.vec(e0 * 2,e0-1)
        assert len(e2) == 2
        assert e3[0].value == 4
        assert e3[1].value == 1
        assert RMExpression.grad(e3, 0, "x") == 2
        assert RMExpression.grad(e3, 1, "x") == 1
