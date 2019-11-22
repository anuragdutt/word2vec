import tensorflow as tf
import numpy as np

def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A =


    And write the code that calculate B = log(sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """

    tw_T = tf.transpose(true_w)
    all_mat = tf.matmul(inputs, tw_T)

    #Each representation of the context word in terms of the predicting elements is given by the diagonal matrix
    # We don't need to take exp and log as log and exponential cancel out

    A = tf.diag_part(all_mat)
    B_exp = tf.exp(all_mat) 
    B_exp_sum = tf.reduce_sum(B_exp, axis = 1)
    B = tf.log(B_exp_sum)

    # print(A.get_shape())
    # print(B.get_shape())
    # exit(0)

    # print("subtract", tf.subtract(B, A).get_shape())
    # exit(0)

    return tf.subtract(B, A)

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """

    # print("inputs:", inputs.get_shape())
    # print("biases:", biases.get_shape())
    # print("labels:", labels.get_shape())
    # print("sample:", sample.shape)
    # print("weights:", weights.get_shape())
    # print("unigram_prob:", len(unigram_prob))

    k = len(sample)
    small_inc = 1e-10

    # ##P(D=1|w) first Term


    ## get the s_o term
    weights_mat = tf.gather(weights, labels)
    weights_mat = tf.reshape(weights_mat, [-1, tf.shape(weights)[1]])
    bias_mat = tf.gather(biases, labels)
    bias_mat = tf.reshape(bias_mat, [-1, 1])
    weights_mat_T = tf.transpose(weights_mat)
    qh_T_qw_full = tf.tensordot(inputs, weights_mat_T, axes = 1)
    qh_T_qw = tf.reshape(tf.diag_part(qh_T_qw_full), [-1, 1])
    s_o = tf.add(qh_T_qw ,bias_mat)


    # get the log term for labels
    uni_prob_labels = tf.gather(np.float32(unigram_prob), labels)
    rhs = tf.scalar_mul(k,uni_prob_labels)
    rhs_log = tf.log(rhs+small_inc)

    ## get the main labels
    positive_term_diff = s_o - rhs_log
    positive_term = tf.log(tf.sigmoid(positive_term_diff)+small_inc)


    ## get the negative sample logistic functions
    weights_mat_sample = tf.gather(weights, sample)
    # weights_mat_sample = tf.reshape(weights_mat_sample, [-1, tf.shape(weights)[1]])
    bias_mat_sample = tf.gather(biases, sample)
    bias_mat_sample = tf.reshape(bias_mat_sample, [-1, 1])
    weights_mat_sample_T = tf.transpose(weights_mat_sample)
    qx_T_qw_full = tf.transpose(tf.tensordot(inputs, weights_mat_sample_T, axes = 1))
    s_x = tf.add(qx_T_qw_full, bias_mat_sample)
    


    ## get the log term for negative samples
    uni_prob_neg = tf.gather(np.asarray(np.float32(unigram_prob)), sample)
    uni_prob_sample = tf.reshape(uni_prob_neg, [-1, 1])
    rhs_sample = tf.scalar_mul(k,uni_prob_sample)
    rhs_log_sample = tf.log(rhs_sample + small_inc)

    ## get the negative samples
    negative_term_diff = tf.transpose(s_x - rhs_log_sample)
    negative_term = tf.sigmoid(negative_term_diff)
    negative_term_compliment = tf.log(1.0 - negative_term + small_inc)
    negative_sampling_term = tf.reduce_sum(negative_term_compliment, 1)


    # exit(0)
    nce = tf.negative(tf.add(positive_term, negative_sampling_term))
    return nce
