import numpy as np

my_array = np.array([1, 2, 3, 4, 5])

print(my_array)

my_array = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])

print(my_array)

a = np.arange(9).reshape((3, 3))
print(a)


my_array[0][-1] = -1
print(my_array)


x = np.empty([3,2], dtype = int)
print(x)