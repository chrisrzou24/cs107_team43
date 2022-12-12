#!/usr/bin/env python3

"""
This module contains our FMExpression class, which supports fowrard-mode automatic differentiation, and supporting methods.
"""
import numpy as np

from .base import Expression

class FMExpression(Expression):
    def __init__(self, value, grad = None):
        """
        Takes in the value of the Expression and optional gradient argument to create a FMExpression object
        """
        super().__init__(value)

        if (isinstance(grad, str)):
            self.grad = {grad: np.array(np.ones(len(self.value)))}
        else:
            self.grad = grad

        self._valid = [int, float, FMExpression]

    def __str__(self):
        return f'The real value is {self.value} and the grad values are {self.grad}'

    def __repr__(self):
        return f'FMExpression({self.value}, {self.grad})'

    def __add__(self, var2):
        """
        Addition function for FMExpression, adds the values and combines the gradient dictionaries
        Args:
            var2: another variable either of type FMExpression, int, or float
        Returns:
            a new FMExpression that represents the added expressions
        """
        if (type(var2) not in self._valid):
            raise TypeError("Needs to be type int, float, or FMExpression")

        new_var = self.from_expression(super().__add__(var2))

        if (type(var2) in [int, float]):
            new_var.grad = self.grad
            return new_var

        new_var.grad = {}
        for key in set(list(self.grad.keys())+list(var2.grad.keys())):
            if key in self.grad and key in var2.grad:
                new_var.grad[key] = np.add(self.grad[key],var2.grad[key])

            elif key in self.grad:
                new_var.grad[key] = self.grad[key]
            else:
                new_var.grad[key] = var2.grad[key]

        return new_var

    def __radd__(self, var2):
        return self.__add__(var2)

    def __sub__(self, var2):
        """
        Subtraction function for FMExpression, subtracts the values and combines the gradient dictionaries
        Args:
            var2: another variable either of type FMExpression, int, or float
        Returns:
            a new FMExpression that represents the subtracted expression
        """
        if (type(var2) not in self._valid):
            raise TypeError("Needs to be type int, float, or FMExpression")

        new_var = self.from_expression(super().__sub__(var2))

        if (type(var2) in [int, float]):
            new_var.grad = self.grad
            return new_var

        new_var.grad = {}
        for key in set(list(self.grad.keys())+list(var2.grad.keys())):
            if key in self.grad and key in var2.grad:
                new_var.grad[key] = np.subtract(self.grad[key],var2.grad[key])

            elif key in self.grad:
                new_var.grad[key] = self.grad[key]
            else:
                new_var.grad[key] = -var2.grad[key]

        return new_var


    def __rsub__(self, var2):
        """
        Reverse subtraction function for FMExpression, in case of something like constant - FMExpression object
        Args:
            var2: another variable either of type int or float
        Returns:
            a new FMExpression that represents the subtracted expression
        """
        if (type(var2) not in self._valid):
            raise TypeError("Needs to be type int, float, or FMExpression")

        new_var = self.from_expression(super().__rsub__(var2))
        new_var.grad = {k: -v for k, v in self.grad.items()}
        
        return new_var

    def __mul__(self, var2):
        """
        Multiplication function for FMExpression, multiplies the values and combines the gradient dictionaries
        Args:
            var2: another variable either of type int or float
        Returns:
            a new FMExpression that represents the multiplied expression
        """
        if (type(var2) not in self._valid):
            raise TypeError("Needs to be type int, float, or FMExpression")

        new_var = self.from_expression(super().__mul__(var2))

        if (type(var2) in [int, float]):
            new_var.grad = {k:v * var2 for k, v in self.grad.items()}
            return new_var

        new_var.grad = {}
        for key in set(list(self.grad.keys())+list(var2.grad.keys())):
            if key in self.grad and key in var2.grad:
                new_var.grad[key] = np.multiply(self.value,var2.grad[key]) + np.multiply(var2.value,self.grad[key])
            elif key in self.grad:
                new_var.grad[key] = np.multiply(var2.value,self.grad[key])
            else:
                new_var.grad[key] = np.multiply(self.value,var2.grad[key])


        return new_var

    def __rmul__(self, var2):
        return self.__mul__(var2)

    def __truediv__(self, var2):
        """
        Division function for FMExpression, divides the values and combines the gradient dictionaries
        Args:
            var2: another variable either of type int or float
        Returns:
            a new FMExpression that represents the divided expression
        """
        if (type(var2) not in self._valid):
            raise TypeError("Needs to be type int, float, or FMExpression")

        new_var = self.from_expression(super().__truediv__(var2))

        if (type(var2) in [int, float]):
            new_var.grad = {k: v / var2 for k, v in self.grad.items()}
            return new_var

        new_var.grad = {}
        for key in set(list(self.grad.keys()) + list(var2.grad.keys())):
            if key in self.grad and key in var2.grad:
                new_var.grad[key] = (np.subtract(np.multiply(self.grad[key], var2.value), np.multiply(var2.grad[key], self.value))) / (np.multiply(var2.value, var2.value))
            elif key in self.grad:
                new_var.grad[key] = np.multiply(self.grad[key], var2.value) / (np.multiply(var2.value, var2.value))
            else:
                new_var.grad[key] = - np.multiply(var2.grad[key], self.value) / (np.multiply(var2.value, var2.value))

        return new_var

    def __rtruediv__(self,var2):
        """
        Reverse division function for FMExpression, in case of something like constant / FMExpression
        Args:
            var2: another variable either of type int or float
        Returns:
            a new FMExpression that represents the divided expression
        """
        if (type(var2) not in self._valid):
            raise TypeError("Needs to be type int, float, or FMExpression")

        new_var = self.from_expression(super().__rtruediv__(var2))

        new_var.grad = {k: - np.divide((var2 * v), (np.multiply(self.value, self.value))) for k, v in self.grad.items()}
        return new_var

    def __pow__(self, var2):
        """
        Power function for FMExpression, powers the values and combines the gradient dictionaries, represents self ** var2
        Args:
            var2: another variable either of type int or float
        Returns:
            a new FMExpression that represents the power expression
        """
        if (type(var2) not in self._valid):
            raise TypeError("Needs to be type int, float, or FMExpression")

        new_var = self.from_expression(super().__pow__(var2))

        if (type(var2) in [int, float]):
            new_var.grad = {k: v * (var2 * self.value ** (var2 - 1)) for k, v in self.grad.items()}
            return new_var

        new_var.grad = {}

        for key in set(list(self.grad.keys())+list(var2.grad.keys())):
            if key in self.grad and key in var2.grad:
                new_var.grad[key] = np.multiply(np.multiply(var2.value, np.power(self.value, (var2.value - 1))), self.grad[key]) + multiply(np.multiply(np.power(self.value,var2. value), np.log(self.value)), var2.grad[key])
            elif key in self.grad:
                new_var.grad[key] = np.multiply(np.multiply(var2.value, np.power(self.value, (var2.value - 1))), self.grad[key])
            else:
                new_var.grad[key] = np.multiply(np.multiply(np.power(self.value, var2.value),np.log(self.value)), var2.grad[key])

        return new_var

    def __neg__(self):
        """
        Negation function for FMExpression
        Args:
            None
        Returns:
            a new FMExpression that represents the negation
        """
        return self.__mul__(-1)

    def exp(self, var2 = None):
        """
        Exponentiation function for FMExpression, exponents the values and combines the gradient dictionaries, represents var2 ** self, reverse of __pow__
        Args:
            var2: another variable either of type int or float
        Returns:
            a new FMExpression that represents the exponented expression
        """
        if (not var2): # we are doing e ** self
            new_var = self.from_expression(super().exp())
            new_var.grad = {k: np.multiply(np.exp(self.value), v) for k, v in self.grad.items()}
            return new_var
        
        new_var = self.from_expression(super().exp(var2))
        new_var.grad = {k: np.multiply(np.multiply((var2 ** self.value), v), np.log(var2)) for k, v in self.grad.items()}
        return new_var

    def sin(self):
        """
        Sin function for FMExpression, takes sin of value and calculates the gradient
        Args:
            None
        Returns:
            a new FMExpression that represents the sin
        """
        new_var = self.from_expression(super().sin())
        new_var.grad = {k: np.multiply(np.cos(self.value), v) for k, v in self.grad.items()}
        return new_var

    def cos(self):
        """
        Cos function for FMExpression, takes cos of value and calculates the gradient
        Args:
            None
        Returns:
            a new FMExpression that represents the cos
        """
        new_var = self.from_expression(super().cos())
        new_var.grad = {k: np.multiply(-np.sin(self.value), v) for k, v in self.grad.items()}
        return new_var

    def tan(self):
        """
        Tan function for FMExpression, takes tan of value and calculates the gradient
        Args:
            None
        Returns:
            a new FMExpression that represents the tan
        """
        new_var = self.from_expression(super().tan())
        new_var.grad = {k: np.divide(v,np.multiply(np.cos(self.value), np.cos(self.value))) for k, v in self.grad.items()}
        return new_var

    def arcsin(self):
        """
        Arcsin function for FMExpression, takes arcsin of value and calculates the gradient
        Args:
            None
        Returns:
            a new FMExpression that represents the arcsin
        """
        new_var = self.from_expression(super().arcsin())
        new_var.grad = {k: np.multiply(1 / np.sqrt(1 - self.value ** 2), v) for k, v in self.grad.items()}
        return new_var

    def arccos(self):
        """
        Arccos function for FMExpression, takes arccos of value and calculates the gradient
        Args:
            None
        Returns:
            a new FMExpression that represents the arccos
        """
        new_var = self.from_expression(super().arccos())
        new_var.grad = {k: np.multiply(-1 / np.sqrt(1 - self.value ** 2), v) for k, v in self.grad.items()}
        return new_var

    def arctan(self):
        """
        Arctan function for FMExpression, takes arctan of value and calculates the gradient
        Args:
            None
        Returns:
            a new FMExpression that represents the arctan
        """
        new_var = self.from_expression(super().arctan())
        new_var.grad = {k: np.multiply(1 / (1 + self.value ** 2), v) for k, v in self.grad.items()}
        return new_var

    def sinh(self):
        """
        Sinh function for FMExpression, takes sinh of value and calculates the gradient
        Args:
            None
        Returns:
            a new FMExpression that represents the sinh
        """
        new_var = self.from_expression(super().sinh())
        new_var.grad = {k: np.multiply(np.cosh(self.value), v) for k, v in self.grad.items()}
        return new_var

    def cosh(self):
        """
        Cosh function for FMExpression, takes cosh of value and calculates the gradient
        Args:
            None
        Returns:
            a new FMExpression that represents the cosh
        """
        new_var = self.from_expression(super().cosh())
        new_var.grad = {k: np.multiply(np.sinh(self.value), v) for k, v in self.grad.items()}
        return new_var

    def tanh(self):
        """
        Tanh function for FMExpression, takes tanh of value and calculates the gradient
        Args:
            None
        Returns:
            a new FMExpression that represents the tanh
        """
        new_var = self.from_expression(super().tanh())
        new_var.grad = {k: np.multiply(1 / np.cosh(self.value) ** 2, v) for k, v in self.grad.items()}
        return new_var

    def sigmoid(self):
        """
        Sigmoid (logistic) function for FMExpression, takes the sigmoid of value and calculates the gradient
        Args:
            None
        Returns:
            a new FMExpression that represents the sigmoid expression
        """
        new_var = self.from_expression(super().sigmoid())
        new_var.grad = {k: np.multiply(np.divide(np.exp(self.value), ((np.exp(self.value) + 1) ** 2)), v) for k, v in self.grad.items()}
        return new_var
        
    def log(self, var2 = None):
        """
        Logarithm function for FMExpression, logs the values and combines the gradient dictionaries, represents log_var2(self)
        Args:
            var2: another variable either of type int or float
        Returns:
            a new FMExpression that represents the logarithm expression
        """
        if (not var2): # we are doing ln(self)
            new_var = self.from_expression(super().log())
            new_var.grad = {k: np.multiplynp.divide(v, self.value) for k, v in self.grad.items()}
            return new_var

        new_var = self.from_expression(super().log(var2))
        new_var.grad = {k: np.multiply(np.divide(v, self.value),(1 / np.log(var2))) for k, v in self.grad.items()}
        return new_var

    def sqrt(self):
        """
        Sqrt function for FMExpression, calls __pow__(0.5)
        Args:
            None
        Returns:
            a new FMExpression that represents the sqrt expression
        """
        return self.__pow__(0.5)

    def value(self, *args):
        """
        Gets the value of a FMExpression object
        Args:
            If none, returns the scalar or vector values stored in the object, if an argument is specified, it returns the value stored at that location
        Returns:
            the value or an array with the values of the FMExpression object
        """
        if (len(args) == 0):
            if (len(self) == 1):
                return self.value[0]
            else:
                return self.value.tolist()
        if (len(args) == 1 and isinstance(args[0], int)):
            return self.value[args[0]]

    def grad(self, *args):
        """
        Gets the gradient of a FMExpression object
        Args:
            either a variable name or a function number and variable name
        Returns:
            the gradient or an array with all the gradients of the RMExpression object
        """
        if (len(args) == 1 and isinstance(args[0], str)):
            return self.grad[args[0]]
        if (len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], str)):
            return self.grad[args[1]][args[0]]

    @staticmethod
    def vec(*args):
        """
        Combines different FMExpressions into a vector to represent vector functions
        Args:
            a list of FMExpressions to be combined into a vector
        Returns:
            A new FMExpression representing the vector of FMExpressions
        """
        vec_value = np.array([])
        vec_grad = {}

        for x in args:
            if (type(x) in [int, float]):
                vec_value = np.concatenate([vec_value, x])
                vec_grad = {k:np.concatenate([v,np.array([0])]) for k, v in vec_grad.items()}
            else:
                vec_value = np.concatenate([vec_value, x.value])
                vec_grad = {key:np.concatenate([vec_grad.get(key,np.array([])), x.grad.get(key, np.array([]))]) for key in set(list(vec_grad.keys()) + list(x.grad.keys()))}
        
        return FMExpression(vec_value, vec_grad)

    @classmethod
    def from_expression(cls, expr):
        """
        Cast an Expression object to FMExpression
        Args:
            Expression type to be cast to FMExpression
        Returns:
            FMExpression variable of Expression object
        """
        return FMExpression(expr.value)