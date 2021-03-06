import keras.backend as K
import tensorflow as tf


def custom_loss(n_bags: int, n_seg: int,
                coef1: float = 0.00008, coef2: float = 0.00008):
    """Wrapper for loss function

    Args:
        n_bags: Number of bags
        n_seg: Number of instances (segments) in each bag
        coef1: Value indicating importance of temporal smoothness constraint term
        coef2: Value indicating importance of sparsity constraint term

    Returns:
        An actual loss function
    """
    NORM_LABEL = 0
    ABNORM_LABEL = 1

    def mil_loss_with_constrains(y_true, y_pred):
        """
        MIL loss function with additional constraints for scores of
        instances in abnormal bags: (1) temporal smoothness and (2) sparsity
        """
        # input predictions and labels are 1D, we want 2D
        y_true = K.reshape(y_true, shape=(n_bags, n_seg))
        y_pred = K.reshape(y_pred, shape=(n_bags, n_seg))

        # bag-level labels are the same as labels of each instance (video segment)
        y_true_bag = y_true[:, 0]

        # max score in each bag is taken as bag-level prediction
        # for hinge loss computation
        max_scores = K.max(y_pred, axis=-1)
        abnorm_mask = K.equal(y_true_bag, ABNORM_LABEL)
        norm_mask = K.equal(y_true_bag, NORM_LABEL)
        abnorm_max_scores = tf.boolean_mask(max_scores, abnorm_mask)
        norm_max_scores = tf.boolean_mask(max_scores, norm_mask)

        # hinge loss pushes scores for abnormal and normal segments far apart
        # i.e. it maximizes difference between abnormal and normal scores
        def partial_loss(norm_max_score):
            return K.sum(K.maximum(0., 1. - abnorm_max_scores + norm_max_score),
                         axis=-1)

        hinge_loss = K.mean(K.map_fn(partial_loss, norm_max_scores))

        # sparsity constraint applied to abnormal bags only
        abnorm_pred = tf.boolean_mask(y_pred, abnorm_mask)
        sparsity_term = K.sum(abnorm_pred)  # sum over all dims (segments and bags)

        # temporal smoothness constraint (L2) applied to abnormal bags only
        tmp_score_diff = abnorm_pred[:, :n_seg - 1] - abnorm_pred[:, 1:]
        smooth_term = K.sum(K.pow(tmp_score_diff, 2))
        # print(f'smooth_term shape: {K.int_shape(smooth_term)}')
        # print(f'sparsity_term shape: {K.int_shape(sparsity_term)}')

        loss = hinge_loss + coef1 * smooth_term + coef2 * sparsity_term
        return loss

    return mil_loss_with_constrains
