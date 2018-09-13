import numpy as np
import matplotlib as mlp
# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
v = np.array([1, 0, 1])
vv = np.tile(v, (4, 1))    # Stack 4 copies of v on top of each other
print(vv)                  # Prints "[[1 0 1]
y = x + vv  # Add x and vv elementwise
print(y)  # Prints "[[ 2  2  4

mlp.style.use('ggplot')
x = np.linspace(0, 10, 500)
x = 2*x

print('Hola mundo')
