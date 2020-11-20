# Finite Element Project 1

All the source code for the project is located in the folder `fem`. The 
supplied code for the mesh generation in the folder `fem/supplied`. The 
folder `tests` contains a few unit-tests for the two of the classes.

The runnable example files are located in `fem/examples`, `poisson.py` solves 
the Poisson problem with f = 1, with Neumann conditions for y > 0, and Dirichlet
for y <= 0. `poisson_error.py` runs the test problem, and plots the error 
compared to the analytical solution. The file `poisson_convergence.py` runs 
the test problem for a few different meshes.


# Finite Element Prosject 2

Here we solved the equation describing linear elasticity. The code for this 
project is located in `fem/examples/elasticity`. Here the program `elasticity
.py` can be run to solve a problem of given size, and the program 
`convergence.py` can be run to solve problems with increasing size.
