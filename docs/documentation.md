# AutoDiff Package Documentation

### Team Members: Chris Zou, John Ho, Sho Sho Ho, Jun Chong, Jerry Yang

## Introduction

Differentiation is a complicated problem for computers. One method for calculating derivatives is numerical differentiation, where computers will use a method like Newton’s method to approximate/estimate the derivative of a function. However, this method isn’t just slow; it also becomes imprecise at higher complexities due to rounding error. Alternatively, computers can also use symbolic differentiation,  where they hand a closed-form function to a symbolic differentiation program. However, this limits differentiations to closed-form expressions, and symbolic differentiation is still quite inefficient. Automatic differentiation solves both of these issues and offers an accurate and fast method to calculate derivatives.

Calculating derivatives is imperative in machine learning and deep learning for the purpose of gradient descent.  In order to minimize the loss of our neural network, we must take the partial derivative of the loss function to understand the optimal way to adjust our weights and biases. Furthermore, differentiation is also imperative in statistical analysis and economics. Thus, automatic differentiation is incredibly useful for a large range of applications in STEM.


## Background

Automatic differentiation first and foremost uses the chain rule to perform every single evaluation in the differentiation. This avoids the problems that arise in numerical and symbolic differentiation, namely approximation/stability issues and complexity/cost issues, respectively. Autodiff breaks the function down into a tree where each node is an operation, or a variable representing some value/computation of values. Each edge (connection between the nodes) represents a location to be differentiated. This differentiation will involve the chain rule. After all the differentiations are completed, the algorithm multiplies all the values together to get a final answer.

The reason that autodiff works lies in the fact that each function is comprised of either binary operations (i.e., addition, subtraction, multiplication, division, etc) or functions (exponentiation, logarithm, trigonometric functions like sine and cosine, etc), all of which the differentiations of are trivial. Thus, we conclude that we almost certainly can use the chain rule to find the differentiation across the entire function.

One particular way to do autodiff is called forward accumulation, and this method uses a concept called dual numbers, something very similar to complex numbers. Every number $z$ is written as $a + b \epsilon$, where $a$ and $b$ are real but $\epsilon$ is a nilpotent number with the property $\epsilon^2 = 0$ but $\epsilon \neq 0$. The number $\epsilon$ is a theoretical number called an infinitesimal, a number that is closer to 0 than any other number (it is worth mentioning that this number is theoretical, in a similar way that $i$ is imaginary, and doesn’t exist in the real numbers).


## How to Use *Autodiff43*

Please visit [this website](https://test.pypi.org/project/Autodiff43/) for more information about the package. All the documentation for our project is located in the docs directory. All the code and logic for our project is located in the package directory, Autodiff43.

Before installing any software, please make sure you have [Python3](https://www.python.org/downloads/) installed on your machine. Next, proceed by creating a virtual environment to run the software and install the dependencies. first create the virtual environment:

```console
foo@bar:~$ python3 -m venv env
```

Next, please activate the virtual environment as follows:

```console
foo@bar:~$ source env/bin/activate
```

Once you have activated the virtual environment, please install the dependencies. To do so, first make sure you are in the root directory of this repository, and then execute the following command:

```console
foo@bar:~$ python3 -m pip install -r requirements.txt
```

Next, install the package:

```
pip install -i https://test.pypi.org/simple/ Autodiff43
```

Once you are finished working with this package, remember to deactivate the environment using:

```console
foo@bar:~$ deactivate
```

Ensure that you reacticate the virtual environment before picking the work back up!

Now, you should have everything you need to get started using the package! Just don't forget to deactivate your virtual environment when you are done, and reactivate when you want to work again.

Here is the basic usage of the package, assuming that the FMExpression and RMExpression classes have been imported:

```python
# Create variable expression with x = 4 and dx/dx = 1
e1 = FMExpression(4, "x")
# Create new expression that is a series of elementary operations on `x`
f1 = FMExpression.log(e1) + FMExpression.sin(e1)

# Get the primal trace result, the function evaluated at x = 4
FMExpression.value(f1)

# Get the tangent trace result, partial derivative evaluated at x = 4
FMExpression.grad(f1)

# Create vector expression
e1 = FMExpression(2, "x")
e2 = FMExpression(3, "y")
f = FMExpression.vec([e1 + e2, e1 * e2])

# Get partial derivative of first function in vector
FMExpression.grad(f, 0, "x")

# Get partial derivative of second function in vector
FMExpression.grad(f, 1, "y")
```

Reverse mode is implemented identically, simply replace `FMExpression` with `RMExpression`

Additionally, the following mathematical operations and functions are also supported:
```python
# Assuming e is either of type FMExpression or RMExpression. Expression is shorthand for either FMExpression of RMExpression
Expression.sin(e)
Expression.cos(e)
Expression.tan(e)
Expression.log(e)
Expression.arcsin(e)
Expression.arccos(e)
Expression.arctan(e)
Expression.sinh(e)
Expression.cosh(e)
Expression.tanh(e)
Expression.sigmoid(e) # the logistic function
Expression.sqrt(e)

# Negation is also supported
-e

# The following assume c is a constant or an Expression.
e + c
c + e
e - c
c - e
e * c
c * c
e / c
c / e
e ** c
c ** e
Expression.log(c, e)
```

## Software Organization

### Directory Structure:

- LICENSE
- README.md
- requirements.txt
- pyproject.toml
- dist
    - package distribution files
- docs
    - future_features.md
    - milestone1
    - milestone2.md
    - milestone2_progress.md
    - documentation.md
    - README.md
- Autodiff43
    - \_\_init__.py 
    - logic
        - \_\_init__.py
        - forward_mode.py
        - reverse_mode.py
        - core.py
        - base.py
        - utils.py
    - test
        - \_\_init__.py
        - test_coverage.py
        - test_expression.py

### Modules:

- dist: files used in the distrubtion of our package
- docs: In the docs directory, you will find documentation for the entire project. This will include documentation for each milestone when we were buiding out this library, as well as our final documentation.
- Autodiff43: this will host the majority of our actual work and code.
    - logic: include forward_mode.py and reverse_mode.py, which provide the logic and coding backbone of running autodiff with forward mode and reverse mode. Necessary companion files are also included
    - test: the two test files include coverage testing as well as testing for our Expression class.


### Test:

Our test suite will be located in the test folder. test_expression.py and test_coverage.py will have code to run tests.

### Package Installation

We will distribute the package called Autodiff43 using PyPI. This is located at https://test.pypi.org/project/Autodiff43/. The command to install is both in the link and also listed above.

## Implementation

In order to implement automatic differentiation, there are many classes and functions that are needed. These need to work together to implement each step of automatic differentiation. To conduct the differentiation evaluation, we first need to implement an Expression class. This Expression class will serve as the central part of our package and will handle all variable expressions required in the automatic differentiation process. The idea of this class is to be modular and contain information about one variable expression, while also being able to combine with other instances of the Expression class.

The Expression class serves as a basis for FMExpression and RMExpression, the latter two inherit from the first. The reasoning is that FMExpression and RMExpression calculate values the exact same way, they simply differ on the process to calculate the dervative. Thus Expression contains all the functionality of add, sub, mul, etc, the same function that FMExpression and RMExpression. FMExpression and RMExpression will have overrides, but they will call the Expression class functions first using `super().function_call()` to calculate the value, then go on to compute whatever additional things they need to do separately.

Each FMExpression object will have `value` and `grad` properties, representing the primal and tangent traces, respectively. Leveraging the properties of dual numbers, we can compute these properties for an intermediate expression from previous variables, i.e. the `value` and `grad` properties of other FMExpression objects. This allows us to track the primal and tangent traces simultaenously through each step of evaluation. To accomodate multivariable functions $\mathbf{f}: \mathbb{R}^{m} \mapsto \mathbb{R}$ and $\mathbf{f}: \mathbb{R}^{m} \mapsto \mathbb{R}^n$, we will represent the `value` and `grad` components as vectors and dictionaries, respectively. The way to create a function or expression in FMExpression that has multiple outputs is using the `FMExpression.vec()`, where the parameters determine the functions. For exmple, `f = FMExpression.vec(e1 + e2, e1 - e2)` will create $f$ as type FMExpression, with the first function $f_1 = e1 + e2$, and second function $f_2 = e1 - e2$. This assume e1 and e2 are also of type FMExpression. For the latter, the $\mathnormal{i}$ th column is the projection of the Jacobian $J$ in the direction of the $i$ th unit vector. We will also allow for assigning a string name for a given direction, and expose getter functions `grad()` and `value()` that allow for flexibile queries, so for example `FMExpression.grad(f, "x")` will return the partial derivative $\frac{\partial f}{\partial x}$. Alternatively, `FMExpression.grad(f, i, "x")` will take the $i$th function of $f$ (if $f$ is a vector), and calculate the dervative with respect to $x$. 

RMExpression functionally works the exact same way. The same functions are implemented, including `vec()` `grad()` and `value()`. In RMExpression, `RMExpression.grad(f, "x")` and `RMExpression.grad(f, i, "x")` do the same thing as FMExpression's functions, giving the gradient along the $x$ variable, and if desired, for a particular function number. RMExpression has vastly different attributes, however. While `value` is the same, `grad` no longer stores a dictionary but a single value, so a `name` variable is introducted to supplement the dictionary aspect. Furthermore, RMExpression also has the `node_edges` attribute, which stores the connections of the graph used to determined the topological sort. Lastly, each node also has a `jacobian` attribute, which stores the jacobian matrix, and is calculated using the `.grad` attribute. The `.grad()` function actually obtains its answers from the `jacobian` attribute.

One external library that we will depend on is NumPy. This will come into play when we are dealing with vector operations and other array manipulations. This implementation of the class-based methods will allow for computations on scalar functions and vector functions. As a result, we can handle the situations of vector functions of vectors and scalar functions of vectors.

## Licensing

We are using the MIT License because it allows for a lot of flexibility in terms of distributing closed sources. It allows us to share code under a copyleft license and is known for being permissive, allowing us and others to use our code for commercial purposes.

## Extension

For our extension, we implemented reverse mode (in addition to the forward mode that we implemented). The implementation details for this can be found in reverse_mode.py in the RMExpression class, which supports reverse-mode automatic differentiation.

## Future Work
There are a lot of exciting future work applicatins that will make our package more useful. Specifically, we believe that automatic differentiation is most useful when developing deep neural networks. While our package makes it possible to develop neural networks, we don't explicitly provide support for developing these neural networks, so users would have to write a substantial amount of code on top of our existing software to make these networks. We believe making an interface to train neural networks will make our package more usable and less tedious work wise. 


## Broader Impact and Inclusitivity
To make it easily accessible to a variety of different groups, we will package our library as a pip module. This will make it easy for people to install and use it. Contributions will happen through GitHub in order to make it as easy for as many people as possible to contribute to the code. We will accept improvements to the library: the process for making changes will be through a pull request; our team will make a commitment to respond to the pull request within two business days. We want to make this library inclusive to individuals of all backgrounds, and this is important because the more people who contribute the more useful this library will be
In terms of barriers, our software is naturally geared towards English speaking individuals with access to technology. Therefore, it would be harder for non-native English speakers or individuals from different countries to use and contribute to our code base. In terms of reviewing pull requests, our team is made up of five individuals, all of whom are East Asian. There may be implicit biases in the way our project was implemented. Moreover, there is only one girl on the team. There may be gendered biases as well in our project that are not clearly discernible. We echo Python’s diversity statement, but we aim to have more concrete actions by making sure that we get someone with a different background than ours to “sign-off” on the decisions we make regarding pull requests. 

