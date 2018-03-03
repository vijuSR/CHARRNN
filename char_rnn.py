import numpy as np


data = open("dinos.txt",'r',encoding="utf-8").read()
data = data.lower()
chars = list(set(data))
data_len, vocab_size = len(data), len(chars)
print("data has %d characters, %d unique" % (data_len, vocab_size))
char_to_ix = {ch: i for i, ch in enumerate(sorted(chars))}
ix_to_char = {i: ch for i, ch in enumerate(sorted(chars))}
print(ix_to_char)
print(char_to_ix)


def initialize_parameters(n_a, n_x, n_y):
    """
    Initialize parameters with small random values

    Returns:
    parameters -- python dictionary containing:
                Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                b --  Bias, numpy array of shape (n_a, 1)
                by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    """
    Wax = np.random.randn(n_a, n_x) * 0.01  # input to hidden
    Waa = np.random.randn(n_a, n_a) * 0.01  # hidden to hidden
    Wya = np.random.randn(n_y, n_a) * 0.01  # hidden to output
    ba = np.zeros((n_a, 1))  # hidden bias
    by = np.zeros((n_y, 1))  # output bias

    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}

    return parameters


def clip(gradients, maxValue):
    '''
    Clips the gradients' values between minimum and maximum.

    Arguments:
    gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
    maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue

    Returns:
    gradients -- a dictionary with the clipped gradients.
    '''

    dWaa, dWax, dWya, dba, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['dba'], gradients[
        'dby']

    # clip to mitigate exploding gradients
    for gradient in [dWax, dWaa, dWya, dba, dby]:
        np.clip(gradient, -maxValue, maxValue, out=gradient)

    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "dba": dba, "dby": dby}

    return gradients


def sample(parameters, char_to_ix, seed):
    """
    Sample a sequence of characters according to a sequence of probability distributions output of the RNN

    Arguments:
    parameters -- python dictionary containing the parameters Waa, Wax, Wya, by, and b.
    char_to_ix -- python dictionary mapping each character to an index.

    Returns:
    indices -- a list of length n containing the indices of the sampled characters.
    """
    Waa, Wax, Wya, by, ba = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['ba']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]

    x = np.zeros((vocab_size, 1))
    a_prev = np.zeros((n_a, 1))

    indices = []

    idx = -1

    # Loop over time-steps t. At each time-step, sample a character from a probability distribution and append
    # its index to "indices". We'll stop if we reach 50 characters (which should be very unlikely with a well
    # trained model), which helps debugging and prevents entering an infinite loop.
    counter = 0
    newline_character = char_to_ix['\n']
    # eol = char_to_ix['.']

    while (idx != newline_character and counter != 50):
        # Step 2: Forward propagate x using the equations (1), (2) and (3)
        a = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, x) + ba)
        z = np.dot(Wya, a) + by
        y = softmax(z)

        # np.random.seed(counter+seed)
        # print(y)
        # Sample the index of a character within the vocabulary from the probability distribution y
        idx = np.random.choice(np.arange(vocab_size), p=y.ravel())

        # Append the index to "indices"
        indices.append(idx)

        # Overwrite the input character as the one corresponding to the sampled index.
        x = np.zeros((vocab_size, 1))
        x[idx] = 1

        # Update "a_prev" to be "a"
        a_prev = a

        # seed += 1
        # counter +=1

    if (counter == 50):
        indices.append(char_to_ix['\n'])

    return indices


def update_parameters(parameters, gradients, lr):
    parameters['Wax'] += -lr * gradients['dWax']
    parameters['Waa'] += -lr * gradients['dWaa']
    parameters['Wya'] += -lr * gradients['dWya']
    parameters['ba']  += -lr * gradients['dba']
    parameters['by']  += -lr * gradients['dby']
    return parameters


def rnn_step_forward(parameters, a_prev, x):
    Waa, Wax, Wya, by, ba = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['ba']
    a_next = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + ba)  # hidden state
    p_t = softmax(
        np.dot(Wya, a_next) + by)  # unnormalized log probabilities for next chars # probabilities for next chars

    return a_next, p_t


def rnn_forward(x, y, a0, parameters, vocab_size=27):
    xs, a, ys = {}, {}, {}
    a[-1] = np.copy(a0)
    loss = 0

    # print(len(a0))
    for t in range(len(x)):
        xs[t] = np.zeros((vocab_size, 1))

        if x[t] != None:
            xs[t][x[t]] = 1
        # print(Wax.shape, xs[t].shape, Waa.shape, a[t-1].shape, ba.shape)
        # print(a[t].shape)
        # print(t, t-1)
        a[t], ys[t] = rnn_step_forward(parameters, a[t - 1], xs[t])
        # print(a[t].shape)

        loss -= np.log(ys[t][y[t], 0])  ##########?????

    cache = (a, xs, ys)
    return loss, cache


def rnn_backward(x, y, cache, parameters):
    gradients = {}
    (a, xs, ys) = cache
    Wax, Waa, Wya, ba, by = parameters["Wax"], parameters["Waa"], parameters["Wya"], parameters["ba"], parameters["by"]
    gradients["dWax"], gradients["dWaa"], gradients["dWya"] = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)
    gradients["dba"], gradients["dby"] = np.zeros_like(ba), np.zeros_like(by)
    gradients["da_next"] = np.zeros_like(a[0])

    for t in reversed(range(len(x))):
        dy = np.copy(ys[t])
        dy[y[t]] -= 1  # backprop into y.
        gradients["dWya"] += np.dot(dy, a[t].T)
        gradients["dby"] += dy
        da = np.dot(Wya.T, dy) + gradients['da_next']
        dhraw = (1 - a[t] * a[t]) * da  # backprop through tanh nonlinearity
        gradients["dba"] += dhraw
        gradients["dWax"] += np.dot(dhraw, xs[t].T)
        gradients["dWaa"] += np.dot(dhraw, a[t - 1].T)
        gradients["da_next"] = np.dot(Waa.T, dhraw)

    return gradients, a


def optimize(x, y, a_prev, parameters, learning_rate=0.01):
    # print(a_prev.shape)
    loss, cache = rnn_forward(x, y, a_prev, parameters)

    # Backpropagate through time
    gradients, a = rnn_backward(x, y, cache, parameters)

    gradients = clip(gradients, 5)

    # Update parameters
    parameters = update_parameters(parameters, gradients, learning_rate)

    return loss, gradients, a[len(x) - 1]


def model(data, ix_to_char, char_to_ix, num_iterations=100000, n_a=50, dino_names=7, vocab_size=27):
    """
    Trains the model and generates dinosaur names.

    Arguments:
    data -- text corpus
    ix_to_char -- dictionary that maps the index to a character
    char_to_ix -- dictionary that maps a character to an index
    num_iterations -- number of iterations to train the model for
    n_a -- number of units of the RNN cell
    dino_names -- number of dinosaur names you want to sample at each iteration.
    vocab_size -- number of unique characters found in the text, size of the vocabulary

    Returns:
    parameters -- learned parameters
    """

    # Retrieve n_x and n_y from vocab_size
    n_x, n_y = vocab_size, vocab_size

    # Initialize parameters
    parameters = initialize_parameters(n_a, n_x, n_y)

    # Initialize loss (this is required because we want to smooth our loss, don't worry about it)
    loss = get_initial_loss(vocab_size, dino_names)

    # Build list of all dinosaur names (training examples).
    with open("dinos.txt", encoding="utf-8") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]

    # Shuffle list of all dinosaur names
    # np.random.seed(0)
    np.random.shuffle(examples)

    # Initialize the hidden state of your LSTM
    a_prev = np.zeros((n_a, 1))

    # Optimization loop
    for j in range(num_iterations):

        # define one training example (X,Y)
        index = j % len(examples)
        X = [None] + [char_to_ix[ch] for ch in examples[index]]
        Y = X[1:] + [char_to_ix["\n"]]

        # Perform one optimization step: Forward-prop -> Backward-prop -> Clip -> Update parameters
        # learning rate of 0.01
        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters)

        # latency trick to keep the loss smooth. It happens here to accelerate the training.
        loss = smooth(loss, curr_loss)

        # Every 2000 Iteration, generate "n" to check if the model is learning properly
        if j % 2000 == 0:

            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')

            # The number of dinosaur names to print
            seed = 0
            for name in range(dino_names):
                # Sample indices and print them
                sampled_indices = sample(parameters, char_to_ix, seed)
                print_sample(sampled_indices, ix_to_char)

                seed += 1  # To get the same result for grading purposed, increment the seed by one.

            print('\n')

    return parameters


def get_initial_loss(vocab_size, seq_length):
    return -np.log(1.0/vocab_size)*seq_length


def smooth(loss, cur_loss):
    return loss * 0.999 + cur_loss * 0.001


def print_sample(sample_ix, ix_to_char):
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    txt = txt[0].upper() + txt[1:]  # capitalize first character
    print ('%s' % (txt, ), end='')


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


model(data, ix_to_char, char_to_ix)
