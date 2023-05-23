# getProbs accepts four parameters:
#
# checkParams: set to true if you want a check on all of the params
#   that makes sure that everything looks OK. This will make the
#   function run slower; use only for debugging
#
# x: 1-D NumPy array, where j^th entry is the number of times that
#   word j was observed in this particular document
#
# pi: the vector of probabilities that tells us how prevalent each doc
#   class is in the corpus
#
# log_allMus: NumPy matrix, where each row is associated with
#   a different document class. A row gives us the list of log-word
#   probabilities associated with the document class.
#
# returns: a NumPy vector, where the j^th entry in the vector is the
#   probability that the document came from each of the different
#   classes
#
def getProbs (checkParams, x, pi, log_allMus):
    #
    if checkParams == True:
            if x.shape [0] != log_allMus.shape [1]:
                    raise Exception ('Number of words in doc does not match')
            if pi.shape [0] != log_allMus.shape [0]:
                    raise Exception ('Number of document classes does not match')
            if not (0.999 <= np.sum (pi) <= 1.001):
                    raise Exception ('Pi is not a proper probability vector')
            for i in range(log_allMus.shape [0]):
                    if not (0.999 <= np.sum (np.exp (log_allMus[i])) <= 1.001):
                            raise Exception ('log_allMus[' + str(i) + '] is not a proper probability vector')
    #
    # to ensure that we don’t have any underflows, we will do
    # all of the arithmetic in “log space”. Specifically, according to
    # the Multinomial distribution, in order to compute
    # Pr[x | class j] we need to compute:
    #
    #       pi[j] * prod_w allMus[j][w]^x[w]
    #
    # If the doc has a lot of words, we can easily underflow. So
    # instead, we compute this as:
    #
    #       log_pi[j] + sum_w x[w] * log_allMus[j][w]
    #
    allProbs = np.log (pi)
    #
    # consider each of the classes, in turn
    for i in range(log_allMus.shape [0]):
            product = np.multiply (x, log_allMus[i])
            allProbs[i] += np.sum (product)
    #
    # so that we don’t have an underflow, we find the largest
    # logProb, and then subtract it from everyone (subtracting
    # from everything in an array of logarithms is like dividing
    # by a constant in an array of “regular” numbers); since we
    # are going to normalize anyway, we can do this with impunity
    #
    biggestLogProb = np.amax (allProbs)
    allProbs -= biggestLogProb
    #
    # finally, get out of log space, and return the answer
    #
    allProbs = np.exp (allProbs)
    return allProbs / np.sum (allProbs)
