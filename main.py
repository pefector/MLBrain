import idx2numpy
from machineLearn3 import *
import matplotlib.pyplot as plt
import json

# Reading
Test_images = idx2numpy.convert_from_file('Test_images.idx3-ubyte')
Test_labels = idx2numpy.convert_from_file('Test_labels.idx1-ubyte')
# Train_labels = idx2numpy.convert_from_file('Train_labels.idx1-ubyte')
# Train_images = idx2numpy.convert_from_file('Train_images.idx3-ubyte')

Test_labels = fix_targets(Test_labels, 10)
# Train_labels = fix_targets(Train_labels, 10)

Test_images = np.array([i.flatten() for i in Test_images])
# Train_images = np.array([i.flatten() for i in Train_images])

parameters = nn_model(Test_images, Test_labels, 10, num_iterations=60, print_cost=True,alpha=0.01,graph=True, Max_last_best=4000)
# model_test(parameters, Test_images, Test_labels)

model_test(parameters, Test_images, Test_labels)
SaveDict(parameters)
plt.show()