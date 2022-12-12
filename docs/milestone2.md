# Milestone 1
### Team Members: Chris Zou, John Ho, Sho Sho Ho, Jun Chong, Jerry Yang

## Introduction

Differentiation is a complicated problem for computers. One method for calculating derivatives is numerical differentiation, where computers will use a method like Newton’s method to approximate/estimate the derivative of a function. However, this method isn’t just slow; it also becomes imprecise at higher complexities due to rounding error. Alternatively, computers can also use symbolic differentiation,  where they hand a closed-form function to a symbolic differentiation program. However, this limits differentiations to closed-form expressions, and symbolic differentiation is still quite inefficient. Automatic differentiation solves both of these issues and offers an accurate and fast method to calculate derivatives.

Calculating derivatives is imperative in machine learning and deep learning for the purpose of gradient descent.  In order to minimize the loss of our neural network, we must take the partial derivative of the loss function to understand the optimal way to adjust our weights and biases. Furthermore, differentiation is also imperative in statistical analysis and economics. Thus, automatic differentiation is incredibly useful for a large range of applications in STEM.


## Background

Automatic differentiation first and foremost uses the chain rule to perform every single evaluation in the differentiation. This avoids the problems that arise in numerical and symbolic differentiation, namely approximation/stability issues and complexity/cost issues, respectively. Autodiff breaks the function down into a tree where each node is an operation, or a variable representing some value/computation of values. Each edge (connection between the nodes) represents a location to be differentiated. This differentiation will involve the chain rule. After all the differentiations are completed, the algorithm multiplies all the values together to get a final answer.

The reason that autodiff works lies in the fact that each function is comprised of either binary operations (i.e., addition, subtraction, multiplication, division, etc) or functions (exponentiation, logarithm, trigonometric functions like sine and cosine, etc), all of which the differentiations of are trivial. Thus, we conclude that we almost certainly can use the chain rule to find the differentiation across the entire function.

One particular way to do autodiff is called forward accumulation, and this method uses a concept called dual numbers, something very similar to complex numbers. Every number $z$ is written as $a + b \epsilon$, where $a$ and $b$ are real but $\epsilon$ is a nilpotent number with the property $\epsilon^2 = 0$ but $\epsilon \neq 0$. The number $\epsilon$ is a theoretical number called an infinitesimal, a number that is closer to 0 than any other number (it is worth mentioning that this number is theoretical, in a similar way that $i$ is imaginary, and doesn’t exist in the real numbers).


## How to Use *Autodiff43*

To use this package, please first install all the code. Please make sure you have Python3 installed on your machine. Once you have downloaded all the code on your machine and are in the correct directory, the first step is to create a virtual environment in order to run the code. You can do that as follows:

```console
foo@bar:~$ python3 -m venv env
```

Next, please activate the virtual environment as follows:

```console
foo@bar:~$ source env/bin/activate
```

Once you are finished working with this package, remember to deactivate the environment using:

```console
foo@bar:~$ deactivate
```

Ensure that you reacticate the virtual environment before picking the work back up!

Once you have activated the virtual environment, please install the dependencies. To do so, first make sure you are in the root directory of this repository, and then execute the following command:

```console
foo@bar:~$ python3 -m pip install -r requirements.txt
```

Now, you should have everything you need to get started using the package! Just don't forget to deactivate your virtual environment when you are done, and reactivate when you want to work again.

Here is the basic usage of the package, assuming that the Expression class has been imported:

```python
# Create variable expression with e = 4 and de/de = 1
e = Expression(4, 1)
# Create new expression that is a series of elementary operations on `e`
f = Expression.log(e) + Expression.sin(e)

# Get the primal trace result, the function evaluated at e=4
f.value()

# Get the tangent trace result, partial derivative evaluated at e=4
f.deriv()

# When we implement multivariable expressions, we expect it'll be like the following
# Create multivariable expression
d = Expression(2, {'d': 1})
d = Expression(4, {'e': 1})
g = d + e
h = ad.Expression([f, g])

# Get all partial derivatives as 2D array
h.deriv()

# Get partial derivative df/de
h.deriv('e')[0]
```

Additionally, the following mathematical operations and functions are also supported:
```python
# Using the same e variable from above
Expression.sin(e)
Expression.cos(e)
Expression.tan(e)
Expression.log(e)

# Negation is also supported
-e

# The following assume c is a constant
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

# Once multivariable expressions are implemented, the above operations all work if c is also of type Expression
```

## Software Organization

### Directory Structure:

- setup.py
- LICENSE
- README.md
- requirements.txt
- docs
    - future_features.md
    - milestone1
    - milestone2.md
    - milestone2_progress.md
- Autodiff43
    - \_\_init__.py 
    - derivative
    	- \_\_init__.py 
        - derivative.py
    - logic
        - \_\_init__.py
        - forward_mode.py
        - reverse_mode.py
    - test
        - \_\_init__.py
        - test_coverage.py
        - test_expression.py

### Modules:

- docs: In the docs directory, you will find documentation for the entire project. This will include documentation for each milestone.
- Autodiff43: this will host the majority of our actual work and code. It will be split into two sections, one to simply calculate derivatives and another to provide the logic for forward and reverse mode.
  - derivative: includes derivative.py which computers derivative.
  - logic: include forward_mode.py and reverse_mode.py, which provide the logic and coding backbone of running autodiff with forward mode and reverse mode.
- test: the two test files include coverage testing as well as testing for our Expression class.

The setup.py file will contain information including: 

* Package name
* Authors
* Where to download
* Other code

### Test:

Our test suite will be located in the test folder. test_expression.py and test_coverage.py will have code to run tests.

### Package Installation

We will distribute the package called Autodiff43 using PyPI. In order to be distributed, the setup.py, README.md and LICENSE files need to be included.

The project is not yet on PyPI. To download and install the project, please download the zip through github. You are also welcome to clone or fork the project.

## Implementation

In order to implement automatic differentiation, there are many classes and functions that are needed. These need to work together to implement each step of automatic differentiation. To conduct the differentiation evaluation, we first need to implement an Expression class. This Expression class will serve as the central part of our package and will handle all variable expressions required in the automatic differentiation process. The idea of this class is to be modular and contain information about one variable expression, while also being able to combine with other instances of the Expression class.

To do this, the Expression class will include elementary operations (in the form of dunder methods) so that each instance of this class can be combined through these elementary operations to form another instance of the Expression class (such as +, -, etc.). However, in cases where the next intermediate expression includes more complex operation–such as Sin(), Cos(), etc–there will be many method overrides within the package that will transform an instance of the Expression class correctly by returning a new instance of the class that has been adjusted depending on the complex operation. 

Each Expression object will have `real` and `dual` properties, representing the primal and tangent traces, respectively. Leveraging the properties of dual numbers, we can compute these properties for an intermediate expression from previous variables, i.e. the `real` and `dual` properties of other Expression objects. This allows us to track the primal and tangent traces simultaenously through each step of evaluation. To accomodate multivariable functions $\mathbf{f}: \mathbb{R}^{m} \mapsto \mathbb{R}$ and $\mathbf{f}: \mathbb{R}^{m} \mapsto \mathbb{R}^n$, we will represent the `real` and `dual` components as vectors and matrices, respectively. For the latter, the $\mathnormal{i}$ th column is the projection of the Jacobian $J$ in the direction of the $i$ th unit vector. We will also allow for assigning a string name for a given direction, and expose getter functions `derivative` and `value` that allow for flexibile queries, so for example `f.derivative("a")` will return the partial derivative $\frac{\partial f}{\partial a}$. 

One external library that we will depend on is NumPy. This will come into play when we are dealing with vector operations and other array manipulations. This implementation of the class-based methods will allow for computations on scalar functions and vector functions. As a result, we can handle the situations of vector functions of vectors and scalar functions of vectors.

## Licensing

We are using the MIT License because it allows for a lot of flexibility in terms of distributing closed sources. It allows us to share code under a copyleft license and is known for being permissive, allowing us and others to use our code for commercial purposes.

## Feedback

### Milestone 1

Feedback received:

Hi, congratulations on completing Milestone1! I have some suggestions and feedback for you. Please address them in the next milestone accordingly.

* The link between func and sol is missing in the example code.

* In the implementation section, we need a more detailed description of the Expression class. Such as, what data structure is supporting the Expression class, what is the tracking mechanism? You can provide pseudocode if it's helpful for your explanation.

Our changes:

* Modified example code for consistent naming.

* Added example for multivariable case to example code.

* Added section on the main properties of the Expression class, and how primal and tangent traces for forward-mode AD are stored and computed in Expression objects.
