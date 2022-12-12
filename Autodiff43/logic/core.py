"""
This module contains the interface that a user of our package will interact with.
"""
from enum import Enum

from .forward_mode import FMExpression
from .reverse_mode import RMExpression

class ADMode(Enum):
	"""Enum for automatic differentiation mode."""
	FORWARD = 1
	REVERSE = 2

	def to_type(self):
		"""Return the class type for objects in current AD mode."""
		if self == ADMode.FORWARD:
			return FMExpression
		elif self == ADMode.REVERSE:
			return RMExpression
		else:
			raise NotImplementedError

	@staticmethod
	def from_str(label):
		if label == "forward":
			return ADMode.FORWARD
		elif label == "reverse":
			return ADMode.REVERSE
		else:
			raise NotImplementedError


### Module Variables
AD_MODE = ADMode.FORWARD # Default is forward-mode AD


def exp(input):
	"""
	Wrapper method for creating FMExpression and RMExpression objects depending
	on AD_MODE.
	
	Args:
		input (any): The input for the expression. Can be an integer, float, numpy type,
		or list of the above.
	Returns:
		Expression: Returns FMExpression or RMExpression object depending on AD_MODE.
	"""
	if AD_MODE == ADMode.FORWARD:
		return FMExpression(input)
	elif AD_MODE == ADMode.REVERSE:
		# TODO: Implement
		# return RMExpression(input)
		return RMExpression(input)
	else:
		raise NotImplementedError


def grad(output, input_list):
	"""
	Computes the partial derivative of the output with respect to each
	variable in input_list. 

	NOTE: We could include a `previous_grad` parameter instead of assuming it's set to 1.
	Args:
		output (Expression): The function to take partial derivatives of.
		input_list (Tuple[Expression]): The variables to take the partial derivative with respect to.
	
	Returns:
		Tuple[np.ndarray]: Tuple where the ith element is the Jacobian of 
		output with respect to the ith element of input_list.
	"""
	if AD_MODE == ADMode.FORWARD:
		# TODO: Support for multivariable expressions. 
		return output.deriv()
	elif AD_MODE == ADMode.REVERSE:
		# TODO: Implement
		# For each element in output, call el.backward() and return the 
		# desired gradient. Then set all the gradients in the graph to zero before
		# next gradient. Ideally this is cached.
		backward(output)
		return [var.grad for var in input_list]
	else:
		raise NotImplementedError


def set_diff_mode(new_mode):
	"""
	Set the mode for automatic differentiation to forward or reverse mode.

	TODO: Refine warning message.

	Args:
		new_mode (str): The mode to set automatic differentiation to.
	Returns:
		None
	"""
	global AD_MODE

	new_mode = ADMode.from_str(new_mode)
	if new_mode != AD_MODE:
		print("Warning: Expressions must have the same AD mode for computing gradients.")
	
	AD_MODE = new_mode
