import numpy as np
import matplotlib.pyplot as plt
import gzip
import pickle
import sys

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def sigmoid_deriv_impl(s):
    return s * (1 - s)

# x = np.arange(-5, 5, 0.5)
#
# plt.plot(x, sigmoid(x))
# plt.plot(x, sigmoid_deriv(x))
# plt.show()

print("loading training data...")

f = gzip.open('../data/mnist.pkl.gz', 'rb')
db = pickle.load(f, encoding='latin1')
f.close()

training_data = db[0]
test_data = db[1]
sample_count = len(training_data[0])
test_count = len(test_data[0])

feature_count = len(training_data[0][0])
assert(feature_count == len(test_data[0][0]))

print("adjusting training data...")

training_input = training_data[0]
training_output = np.zeros((sample_count, 10))

for i in range(0, sample_count):
    digit = training_data[1][i]
    training_output[i][digit] = 1

print("adjusting test data...")

test_input = test_data[0]
test_output = np.zeros((sample_count, 10))

for i in range(0, test_count):
    digit = test_data[1][i]
    test_output[i][digit] = 1

# training_input = np.array([
#     [0, 0, 0, 1],
#     [0, 0, 1, 1],
#     [0, 1, 0, 1],
#     [0, 1, 1, 1],
#     [1, 0, 0, 1],
#     [1, 0, 1, 1],
#     [1, 1, 0, 1],
#     [1, 1, 1, 1],
#     ])
#
# training_output = np.array([
#     [0, 0],
#     [0, 1],
#     [0, 0],
#     [0, 1],
#     [1, 0],
#     [1, 1],
#     [1, 0],
#     [1, 1],
#     ])
#
# sample_count = len(training_input)
# feature_count = len(training_input[0])

epoch_count = 50
sgd_batch_size = 10
input_neuron_count = len(training_input[0])
hidden_neuron_count = 30  # input_neuron_count
output_neuron_count = len(training_output[0])
training_rate = 3.0

#weight_1 = np.random.randn(input_neuron_count, hidden_neuron_count)
#bias_1 = np.random.randn(hidden_neuron_count)
#weight_2 = np.random.randn(hidden_neuron_count, output_neuron_count)
#bias_2 = np.random.randn(output_neuron_count)

################################################################################
#
#  wtf is here:
#
#  changing between 1 and 2 shouldn't change anything -- but it does!
#
################################################################################

# 1

weight_1 = np.zeros((hidden_neuron_count, input_neuron_count))
weight_2 = np.zeros((output_neuron_count, hidden_neuron_count))
bias_1 = np.zeros((hidden_neuron_count, 1))
bias_2 = np.zeros((output_neuron_count, 1))

# 2

# weight_1 = np.zeros((input_neuron_count, hidden_neuron_count)).transpose()
# weight_2 = np.zeros((hidden_neuron_count, output_neuron_count)).transpose()
# bias_1 = np.zeros((1, hidden_neuron_count)).transpose()
# bias_2 = np.zeros((1, output_neuron_count)).transpose()

################################################################################

print("weight_1.shape:", weight_1.shape)
print("weight_2.shape:", weight_2.shape)
print("bias_1.shape:", bias_1.shape)
print("bias_2.shape:", bias_2.shape)

def estimate_quality(input, output):
    weighted_input_hidden_layer = np.dot(weight_1, input) + bias_1
    hidden_layer = sigmoid(weighted_input_hidden_layer)

    weighted_input_output_layer = np.dot(weight_2, hidden_layer) + bias_2
    output_layer = sigmoid(weighted_input_output_layer)

    count = len(input)
    correct_count = 0

    for i in range(0, count):
        output_digit = np.argmax(output_layer[i])
        training_digit = np.argmax(output[i])

        if output_digit == training_digit:
            correct_count += 1

    print ("{0}/{1} correct {2}%".format(correct_count, count, correct_count * 100 / count))


sample_idx_table = list(range(0, sample_count))

print("training...")
np.set_printoptions(threshold=np.nan)

for i in range(0, epoch_count):
    print("epoch", i)

    # np.random.shuffle(sample_idx_table)

    k = 0

    for j in range(0, sample_count, sgd_batch_size):
        idx = sample_idx_table[j : j + sgd_batch_size]

        weight_1_grad = np.zeros(weight_1.shape)
        weight_2_grad = np.zeros(weight_2.shape)
        bias_1_grad = np.zeros(bias_1.shape)
        bias_2_grad = np.zeros(bias_2.shape)

        for l in idx:
            # input = np.reshape(training_input[l], (1, feature_count))
            # output = np.reshape(training_output[l], (1, output_neuron_count))
            input = np.reshape(training_input[l], (feature_count, 1))
            output = np.reshape(training_output[l], (output_neuron_count, 1))

            # feed forward

            # weighted_input_hidden_layer = np.dot(input, weight_1) + bias_1
            weighted_input_hidden_layer = np.dot(weight_1, input) + bias_1
            hidden_layer = sigmoid(weighted_input_hidden_layer)

            # weighted_input_output_layer = np.dot(hidden_layer, weight_2) + bias_2
            weighted_input_output_layer = np.dot(weight_2, hidden_layer) + bias_2
            output_layer = sigmoid(weighted_input_output_layer)

            # calc derivative

            t = (output_layer - output) * sigmoid_deriv_impl(output_layer)

            bias2_deriv = t
            # weight2_deriv = np.dot(hidden_layer.T, t)
            weight2_deriv = np.dot(t, hidden_layer.T)

            # t = np.dot(t, weight_2.T) * sigmoid_deriv_impl(hidden_layer)
            t = np.dot(weight_2.T, t) * sigmoid_deriv_impl(hidden_layer)

            bias1_deriv = t
            # weight1_deriv = np.dot(input.T, t)
            weight1_deriv = np.dot(t, input.T)

            weight_1_grad += weight1_deriv
            weight_2_grad += weight2_deriv
            bias_1_grad += bias1_deriv
            bias_2_grad += bias2_deriv

        # gradient descent

        weight_1 -= training_rate / sgd_batch_size * weight_1_grad
        bias_1 -= training_rate / sgd_batch_size * bias_1_grad
        weight_2 -= training_rate / sgd_batch_size * weight_2_grad
        bias_2 -= training_rate / sgd_batch_size * bias_2_grad

        k += 1

        if k % 1000 == 0:
            # print("weight_1")
            # print(weight_1)
            # print("weight_2")
            # print(weight_2)
            print("bias_1")
            print(bias_1)
            print("bias_2")
            print(bias_2)
            print("--- k:", k)
            sys.exit()

    estimate_quality(test_input, test_output)

print("done.")


#print("weight_1:\n", weight_1)
#print("weight_2:\n", weight_2)
#print("prediction:\n", output_layer)

# sys.exit()



# for i in range(0, 5):
#     image = training_data[0][i].reshape((28, 28))
#     digit = training_data[1][i]
#     plt.title(digit)
#     plt.imshow(image, cmap='Greys')
#     plt.show()

