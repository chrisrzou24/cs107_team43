#!/usr/bin/env python3

import numpy as np

class Expression:
    """Base class for forward-mode and reverse-mode implementions."""
    valid_scalar_types = (int, float, np.int64) # TODO: Include numpy types
    _node_count = 0 # Track the number of nodes created, used in identifier
    leafs = {}

    def __init__(self, value):
        # Unpack value, which can be scalars or Expressions, into numpy array, e.g. [1, x, x+1]
        # TODO: Better handling of scalar vs. iterable types
        if (self.is_valid_scalar(value)):
            self.value = np.array([value]) # Lazy way of ensuring that input is a list
        else:
            new_vals = []
            for f in value:
                # If value is anvar2 Expression, make sure that it is one-dimensional.
                if (isinstance(f, type(self))):
                    if (len(f.value) > 1):
                        raise TypeError("Cannot nest vectors in expression.")
                    new_vals.append(f.value[0])
                elif (self.is_valid_scalar(f)):
                    new_vals.append(f)

                else:
                    raise TypeError(f"The type {type(f)} is not supported.")

            self.value = np.array(new_vals)

        # Create unique identifier for object
        self.id = 'v' + str(Expression._node_count)
        
        Expression._node_count += 1

    def __str__(self):
        return f'The real value is {self.value} and the grad values are {self.grad}'

    def __repr__(self):
        return f'Expression({self.value}, {self.grad})'

    def __len__(self):
        return len(self.value)

    def __add__(self, var2):
        """
        Base addition function for Expression, adds the values
        Args:
            var2: another variable either of type Expression, int, or float
        Returns:
            a new Expression with the added value
        """
        # If var2 is a scalar
        if (self.is_valid_scalar(var2)):
            new_val = self.value + var2
        # If var2 is an Expression
        elif (isinstance(var2, Expression)):
            if (len(self.value) == len(var2.value)):
                new_val = self.value + var2.value
            else:
                raise ValueError("Expressions must be vectors of the same length.")
        else:
                raise ValueError("Needs to be type int, float, or Expression")
        
        return Expression(new_val)

    def __radd__(self, var2):
        return self._add__(var2)

    def __sub__(self, var2):
        """
        Base subtraction function for Expression, subtracts the values
        Args:
            var2: another variable either of type Expression, int, or float
        Returns:
            a new Expression with the subtracted value
        """
        # If var2 is a scalar
        if (self.is_valid_scalar(var2)):
            new_val = self.value - var2
        # If var2 is an Expression
        elif (isinstance(var2, Expression)):
            if (len(self.value) == len(var2.value)):
                new_val = self.value - var2.value
            else:
                raise ValueError("Expressions must be vectors of the same length.")
        else:
                raise ValueError("Needs to be type int, float, or Expression")
        
        return Expression(new_val)

    def __rsub__(self, var2):
        """
        Base reverse subtraction function for Expression, in case of constant - Expression object
        Args:
            var2: another variable either of type Expression, int, or float
        Returns:
            a new Expression with the subtracted value
        """
        # If var2 is a scalar
        if (self.is_valid_scalar(var2)):
            new_val = var2 - self.value
        else:
            raise ValueError("Needs to be type int, float, or Expression")
        
        return Expression(new_val)

    def __mul__(self, var2):
        """
        Base multiplication function for Expression, multiplies the values
        Args:
            var2: another variable either of type Expression, int, or float
        Returns:
            a new Expression with the multiplied value
        """
        # If var2 is a scalar
        if (self.is_valid_scalar(var2)):
            new_val = self.value * var2
        # If var2 is an Expression
        elif (isinstance(var2, Expression)):
            if (len(self.value) == len(var2.value)):
                new_val = self.value * var2.value
            else:
                raise ValueError("Expressions must be vectors of the same length.")
        else:
                raise ValueError("Needs to be type int, float, or Expression")
        
        return Expression(new_val)

    def __rmul__(self, var2):
        return self._mul__(var2)

    def __truediv__(self, var2):
        """
        Base division function for Expression, divides the values
        Args:
            var2: another variable either of type Expression, int, or float
        Returns:
            a new Expression with the subtracted value
        """
        # If var2 is a scalar
        if (self.is_valid_scalar(var2)):
            if (var2 == 0):
                raise ZeroDivisionError
            new_val = self.value / var2
        # If var2 is an Expression
        elif (isinstance(var2, Expression)):
            if (len(self.value) == len(var2.value)):
                if (var2.value == 0):
                    raise ZeroDivisionError

                new_val = self.value / var2.value
            else:
                raise ValueError("Expressions must be vectors of the same length.")
        else:
                raise ValueError("Needs to be type int, float, or Expression")
        
        return Expression(new_val)

    def __rtruediv__(self, var2):
        """
        Base reverse division function for Expression, in case of constant / Expression object
        Args:
            var2: another variable either of type Expression, int, or float
        Returns:
            a new Expression with the divided value
        """

        if (self.value == 0):
            raise ZeroDivisionError
        # If var2 is a scalar
        if (self.is_valid_scalar(var2)):
            new_val = var2 / self.value
        else:
            raise ValueError("Needs to be type int, float, or Expression")
        
        return Expression(new_val)

    def __pow__(self, var2):
        """
        Base power function for Expression, power the values, represents self ** var2
        Args:
            var2: another variable either of type Expression, int, or float
        Returns:
            a new Expression with the power value
        """
        # If var2 is a scalar
        if (self.is_valid_scalar(var2)):
            new_val = self.value ** var2
        # If var2 is an Expression
        elif (isinstance(var2, Expression)):
            if (len(self.value) == len(var2.value)):
                new_val = self.value ** var2.value
            else:
                raise ValueError("Expressions must be vectors of the same length.")
        else:
                raise ValueError("Needs to be type int, float, or Expression")
        
        return Expression(new_val)

    def __neg__(self):
        """
        Base negation function for Expression, negates the value
        Args:
            None
        Returns:
            a new Expression with the negated value
        """
        new_val = self.value * -1
        return Expression(new_val)

    def exp(self, var2 = None):
        """
        Base exponentiation function for Expression, exponent the values, represents var2 ** self
        Args:
            var2: another variable either of type Expression, int, or float
        Returns:
            a new Expression with the exponented value
        """
        if (var2 == None): # we are doing e ** self
            new_val = np.exp(self.value)
        # If var2 is a scalar
        elif (self.is_valid_scalar(var2)):
            new_val = var2 ** self.value
        elif (isinstance(var2, Expression)):
            if (len(self.value) == len(var2.value)):
                new_val = var2.value ** self.value
            else:
                raise ValueError("Expressions must be vectors of the same length.")
        else:
                raise ValueError("Needs to be type int, float, or Expression")

        return Expression(new_val)

    def sin(self):
        """
        Base sin function for Expression, sines the value
        Args:
            None
        Returns:
            a new Expression with the sin value
        """
        new_val = np.sin(self.value)
        return Expression(new_val)

    def cos(self):
        """
        Base cos function for Expression, cosines the value
        Args:
            None
        Returns:
            a new Expression with the cos value
        """
        new_val = np.cos(self.value)
        return Expression(new_val)

    def tan(self):
        """
        Base tan function for Expression, tans the value
        Args:
            None
        Returns:
            a new Expression with the tan value
        """
        new_val = np.tan(self.value)
        return Expression(new_val)

    def arcsin(self):
        """
        Base arcsin function for Expression, arcsines the value
        Args:
            None
        Returns:
            a new Expression with the arcsin value
        """
        new_val = np.arcsin(self.value)
        return Expression(new_val)

    def arccos(self):
        """
        Base arccos function for Expression, arccossines the value
        Args:
            None
        Returns:
            a new Expression with the arccos value
        """
        new_val = np.arccos(self.value)
        return Expression(new_val)

    def arctan(self):
        """
        Base arctan function for Expression, arctans the value
        Args:
            None
        Returns:
            a new Expression with the arctan value
        """
        new_val = np.arctan(self.value)
        return Expression(new_val)

    def sinh(self):
        """
        Base sinh function for Expression, sinh the value
        Args:
            None
        Returns:
            a new Expression with the sinh value
        """
        new_val = np.sinh(self.value)
        return Expression(new_val)

    def cosh(self):
        """
        Base cosh function for Expression, cosh the value
        Args:
            None
        Returns:
            a new Expression with the cosh value
        """
        new_val = np.cosh(self.value)
        return Expression(new_val)

    def tanh(self):
        """
        Base tanh function for Expression, tanh the value
        Args:
            None
        Returns:
            a new Expression with the tanh value
        """
        new_val = np.tanh(self.value)
        return Expression(new_val)

    def sigmoid(self):
        """
        Base sigmoid (logistic) function for Expression, sigmoid the value
        Args:
            None
        Returns:
            a new Expression with the sigmoid value
        """
        new_val = 1 / (1 + np.exp(-self.value))
        return Expression(new_val)

    def log(self, var2 = None):
        """
        Base logarithm function for Expression, logarithm the values, represents log_var2(self)
        Args:
            var2: another variable either of type Expression, int, or float
        Returns:
            a new Expression with the logarithm value
        """
        if (var2 == None): # we are doing ln(self)
            new_val = np.log(self.value)
        # If var2 is a scalar
        elif (self.is_valid_scalar(var2)):
            new_val = np.log(self.value) / np.log(var2)
        elif (isinstance(var2, Expression)):
            if (len(self.value) == len(var2.value)):
                new_val = np.log(self.value) / np.log(var2.value)
            else:
                raise ValueError("Expressions must be vectors of the same length.")
        else:
                raise ValueError("Needs to be type int, float, or Expression")

        return Expression(new_val)

    def sqrt(self):
        """
        Base sqrt function for Expression, calls __pow__(0.5)
        Args:
            None
        Returns:
            a new Expression with the sqrt value
        """
        return self.__pow__(0.5)

    @classmethod
    def is_valid_scalar(cls, value):
        return isinstance(value, cls.valid_scalar_types)
