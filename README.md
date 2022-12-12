# Team 43 Project Repository

![test.yml](https://github.com/chrisrzou24/cs107_team43/actions/workflows/test.yml/badge.svg)
![coverage.yml](https://github.com/chrisrzou24/cs107_team43/actions/workflows/coverage.yml/badge.svg)

Authored by Christopher Zou, John Ho, Jun Chong, Jerry Yang, Sho Sho Ho.

Please view https://test.pypi.org/project/Autodiff43/ for information about our package.

This package can be installed using the following command

`pip install -i https://test.pypi.org/simple/ Autodiff43==X.Y.Z`

Where `X.Y.Z` is the latest version of the package.

Please install the dependencies. To do so, first make sure you are in the root directory of this repository, and then execute the following command:

`python3 -m pip install -r requirements.txt`

Once the package has been installed, the following commands are necessary to use our package:

`from Autodiff43.logic.forward_mode import FMExpression`

`from Autodiff43.logic.reverse_mode import RMExpression`

These two import lines will provide the functionality to create `FMExpression` and `RMExpression` objects, which drive forward mode autodiff and reverse mode autodiff, respectively.


All the documentation for our project is located in the docs directory, which will also contain further information about how to use.
All the code and logic for our project is located in the package directory, Autodiff43.