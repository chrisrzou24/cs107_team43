#!/usr/bin/env python3

"""
This module contains our RMExpression class, which supports reverse-mode automatic differentiation, and supporting methods.
"""
import numpy as np

from .base import Expression
from .utils import topological_sort, clear_grad

class RMExpression(Expression):
    def __init__(self, value, name = None, node_edges = None):
        """
        Takes in the value of the Expression and optional name and node_edges arguments to create a RMExpression object
        """
        super().__init__(value)
        self.name = name
        if (not node_edges):
            self.node_edges = [] # Output nodes are root
        else:
            self.node_edges = node_edges

        self.grad = np.array([0] * len(self.value)) # Always initialized to 0 before backward pass.
        self.jacobian = None

        self._valid = [int, float, RMExpression]
    
    def __str__(self):
        return f'Name: {self.name} has a real value of {self.value} and the grad values are {self.grad}. Node_edges are {self.node_edges} and leaves are {self.leaf}.'

    def __add__(self, var2):
        """
        Addition function for RMExpression, adds the values and updates the node_edges
        Args:
            var2: another variable either of type RMExpression, int, or float
        Returns:
            a new RMExpression that represents the added expressions
        """
        if (type(var2) not in self._valid):
            raise TypeError("Needs to be type int, float, or RMExpression")

        new_var = self.from_expression(super().__add__(var2))

        if (type(var2) in [int, float]):
            new_var.node_edges.append((self, 1))
            return new_var

        new_var.node_edges.append((self, 1))
        new_var.node_edges.append((var2, 1))

        return new_var

    def __radd__(self, var2):
        return self.__add__(var2)

    def __sub__(self, var2):
        """
        Subtraction function for RMExpression, subtracts the values and updates the node_edges
        Args:
            var2: another variable either of type RMExpression, int, or float
        Returns:
            a new RMExpression that represents the subtracted expressions
        """
        if (type(var2) not in self._valid):
            raise TypeError("Needs to be type int, float, or RMExpression")

        new_var = self.from_expression(super().__sub__(var2))

        if (type(var2) in [int, float]):
            new_var.node_edges.append((self, 1))
            return new_var

        new_var.node_edges.append((self, 1))
        new_var.node_edges.append((var2, -1))

        return new_var

    def __rsub__(self, var2):
        """
        Reverse subtraction function for RMExpression, in case of something like constant - RMExpression object
        Args:
            var2: another variable either of type RMExpression, int, or float
        Returns:
            a new RMExpression that represents the subtracted expressions
        """
        if (type(var2) not in self._valid):
            raise TypeError("Needs to be type int, float, or RMExpression")

        new_var = self.from_expression(super().__rsub__(var2))
        new_var.node_edges.append((self, -1))
        return new_var

    def __mul__(self, var2):
        """
        Multiplication function for RMExpression, multiplies the values and updates the node_edges
        Args:
            var2: another variable either of type RMExpression, int, or float
        Returns:
            a new RMExpression that represents the multiplied expressions
        """
        if (type(var2) not in self._valid):
            raise TypeError("Needs to be type int, float, or RMExpression")

        new_var = self.from_expression(super().__mul__(var2))

        if (type(var2) in [int, float]):
            new_var.node_edges.append((self, var2))
            return new_var

        new_var.node_edges.append((self, var2.value))
        new_var.node_edges.append((var2, self.value))

        return new_var

    def __rmul__(self, var2):
        return self.__mul__(var2)

    def __truediv__(self, var2):
        """
        Division function for RMExpression, divides the values and updates the node_edges
        Args:
            var2: another variable either of type RMExpression, int, or float
        Returns:
            a new RMExpression that represents the divided expressions
        """
        if (type(var2) not in self._valid):
            raise TypeError("Needs to be type int, float, or RMExpression")

        new_var = self.from_expression(super().__truediv__(var2))

        if (type(var2) in [int, float]):
            new_var.node_edges.append((self, 1 / var2))
            return new_var

        new_var.node_edges.append((self, 1 / var2.value))
        new_var.node_edges.append((var2, -self.value / var2.value ** 2))

        return new_var

    def __rtruediv__(self, var2):
        """
        Reverse division function for RMExpression, in case of something like constant / RMExpression object
        Args:
            var2: another variable either of type RMExpression, int, or float
        Returns:
            a new RMExpression that represents the divided expressions
        """
        if (type(var2) not in self._valid):
            raise TypeError("Needs to be type int, float, or FMExpression")

        new_var = self.from_expression(super().__rtruediv__(var2))
        new_var.node_edges.append((self, -var2 / self.value ** 2))
        return new_var

    def __pow__(self, var2):
        """
        Power function for RMExpression, powers the values and updates the node_edges, represents self ** var2
        Args:
            var2: another variable either of type RMExpression, int, or float
        Returns:
            a new RMExpression that represents the power expressions
        """
        if (type(var2) not in self._valid):
            raise TypeError("Needs to be type int, float, or RMExpression")

        new_var = self.from_expression(super().__pow__(var2))

        if (type(var2) in [int, float]):
            new_var.node_edges.append((self, var2 * self.value ** (var2 - 1)))
            return new_var

        new_var.node_edges.append((self, var2.value * self.value ** (var2.value - 1)))
        new_var.node_edges.append((var2, self.value ** var2.value * np.log(self.value)))

        return new_var

    def __neg__(self):
        """
        Negation function for RMExpression
        Args:
            None
        Returns:
            a new RMExpression that represents the negation
        """
        new_var = self.from_expression(super().__neg__())
        new_var.node_edges.append((self, -1))
        return new_var

    def exp(self, var2 = None):
        """
        Exponentiation function for RMExpression, exponents the values and updates the node_edges, represents var2 ** self, reverse of __pow__
        Args:
            var2: another variable either of type int or float
        Returns:
            a new RMExpression that represents the exponented expression
        """
        if (var2 == None): # we are doing e ** self
            new_var = self.from_expression(super().exp())
            new_var.node_edges.append((self, np.exp(self.value)))
            return new_var

        new_var = self.from_expression(super().exp(var2))

        if (type(var2) in [int, float]):
            new_var.node_edges.append((self, var2 ** self.value * np.log(var2)))
            return new_var

        new_var.node_edges.append((self, var2.value ** self.value * np.log(var2.value)))
        new_var.node_edges.append((var2, self.value * var2.value ** (self.value - 1)))

        return new_var

    def sin(self):
        """
        Sin function for RMExpression, takes sin of value and adds to node_edges
        Args:
            None
        Returns:
            a new RMExpression that represents the sin
        """
        new_var = self.from_expression(super().sin())
        new_var.node_edges.append((self, np.cos(self.value)))
        return new_var

    def cos(self):
        """
        Cos function for RMExpression, takes cos of value and adds to node_edges
        Args:
            None
        Returns:
            a new RMExpression that represents the cos
        """
        new_var = self.from_expression(super().cos())
        new_var.node_edges.append((self, -np.sin(self.value)))
        return new_var

    def tan(self):
        """
        Tan function for RMExpression, takes tan of value and adds to node_edges
        Args:
            None
        Returns:
            a new RMExpression that represents the tan
        """
        new_var = self.from_expression(super().tan())
        new_var.node_edges.append((self, 1 / np.cos(self.value) ** 2))
        return new_var

    def arcsin(self):
        """
        Arcsin function for RMExpression, takes arcsin of value and adds to node_edges
        Args:
            None
        Returns:
            a new RMExpression that represents the arcsin
        """
        new_var = self.from_expression(super().arcsin())
        new_var.node_edges.append((self, 1 / (np.sqrt(1 - self.value ** 2))))
        return new_var

    def arccos(self):
        """
        Arccos function for RMExpression, takes arccos of value and adds to node_edges
        Args:
            None
        Returns:
            a new RMExpression that represents the arccos
        """
        new_var = self.from_expression(super().arccos())
        new_var.node_edges.append((self, -1 / (np.sqrt(1 - self.value ** 2))))
        return new_var

    def arctan(self):
        """
        Arctan function for RMExpression, takes arctan of value and adds to node_edges
        Args:
            None
        Returns:
            a new RMExpression that represents the arctan
        """
        new_var = self.from_expression(super().arctan())
        new_var.node_edges.append((self, 1 / (1 + self.value ** 2)))
        return new_var

    def sinh(self):
        """
        Sinh function for RMExpression, takes sinh of value and adds to node_edges
        Args:
            None
        Returns:
            a new RMExpression that represents the sinh
        """
        new_var = self.from_expression(super().sinh())
        new_var.node_edges.append((self, np.cosh(self.value)))
        return new_var

    def cosh(self):
        """
        Cosh function for RMExpression, takes cosh of value and adds to node_edges
        Args:
            None
        Returns:
            a new RMExpression that represents the cosh
        """
        new_var = self.from_expression(super().cosh())
        new_var.node_edges.append((self, np.sinh(self.value)))
        return new_var

    def tanh(self):
        """
        Tanh function for RMExpression, takes tanh of value and adds to node_edges
        Args:
            None
        Returns:
            a new RMExpression that represents the tanh
        """
        new_var = self.from_expression(super().tanh())
        new_var.node_edges.append((self, 1 / np.cosh(self.value) ** 2))
        return new_var

    def sigmoid(self):
        """
        Sigmoid (logistic) function for RMExpression, takes the sigmoid of value and updates the node_edges
        Args:
            None
        Returns:
            a new RMExpression that represents the sigmoid expression
        """
        new_var = self.from_expression(super().sigmoid())
        new_var.node_edges.append((self, np.exp(-self.value) / (np.exp(-self.value) + 1) ** 2))
        return new_var

    def log(self, var2 = None):
        """
        Logarithm function for RMExpression, logs the values and combines the updates the node_edges, represents log_var2(self)
        Args:
            var2: another variable either of type int or float
        Returns:
            a new RMExpression that represents the logarithm expression
        """
        if (not var2): # we are doing ln(self)
            new_var = self.from_expression(super().log())
            new_var.node_edges.append((self, 1 / self.value))
            return new_var

        new_var = self.from_expression(super().log(var2))

        if (type(var2) in [int, float]):
            new_var.node_edges.append((self, 1 / (self.value * np.log(var2))))
            return new_var

        new_var.node_edges.append((self, 1 / (self.value * np.log(var2.value))))
        new_var.node_edges.append((var2, -np.log(self.value) / (var2.value * np.log(var2.value) ** 2)))

        return new_var

    def sqrt(self):
        """
        Sqrt function for RMExpression, calls __pow__(0.5)
        Args:
            None
        Returns:
            a new FMExpression that represents the sqrt expression
        """
        return self.__pow__(0.5)

    def backward_scalar(self):
        """
        Computes the partial derivative of self (parent node) with respect to each leaf node by updating the gradient at every node.
        Proceeds in order of topological sort, updating the .grad attribute at each point, and updating self.jacobian when a variable RMExpression is reached
        Args:
            None
        Returns:
            None
        """
        self.jacobian = dict()

        # clears all gradients
        clear_grad(self)
        self.grad = 1

        topo_sort = topological_sort(self)
        for node in topo_sort:

            if (len(node.node_edges) == 0):
                name = node.name
                self.jacobian[name] = node.grad

            for (child, edge_weight) in node.node_edges:
                child.grad = child.grad + edge_weight * node.grad

    def value(self, *args):
        """
        Gets the value of a RMExpression object
        Args:
            If none, returns the scalar or vector values stored in the object, if an argument is specified, it returns the value stored at that location
        Returns:
            the value or an array with the values of the RMExpression object
        """
        if (len(args) == 0):
            if (len(self) == 1):
                return self.value[0]
            else:
                ret = []
                for i in range(len(self)):
                    ret.append(self[i].value[0])
                return ret
        if (len(args) == 1):
            return self[args[0]].value[0]

    def grad(self, *args):
        """
        Gets the gradient of a RMExpression object
        Args:
            either a variable name or a function number and variable name
        Returns:
            the gradient or an array with all the gradients of the RMExpression object
        """
        if (len(args) == 1 and isinstance(args[0], str)):
            if (len(self) > 1):
                for i in range(len(self)):
                    if (self[i].jacobian == None):
                        self[i].backward_scalar()

                ret = np.array([])
                for i in range(len(self)):
                    ret = np.concatenate([ret, [self[i].jacobian[args[0]]]])

                return ret

            if (self.jacobian == None):
                self.backward_scalar()
            return self.jacobian[args[0]]

        if (len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], str)):
            for i in range(len(self)):
                if (i == args[0]):
                    if (self[i].jacobian == None):
                        self[i].backward_scalar()
                    return self[i].jacobian[args[1]]

    @staticmethod
    def vec(*args):
        """
        Combines different RMExpressions into a vector to represent vector functions
        Args:
            a list of RMExpressions to be combined into a vector
        Returns:
            A vector of RMExpressions
        """
        exprs = []
        for func in args:
            exprs.append(func)
        return exprs

    @classmethod
    def from_expression(cls, expr):
        """
        Cast an Expression object to RMExpression
        Args:
            Expression type to be cast to RMExpression
        Returns:
            RMExpression variable of Expression object
        """
        return RMExpression(expr.value)
