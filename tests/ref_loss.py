# reference implementation of loss from:
# https://github.com/WaqasSultani/AnomalyDetectionCVPR2018/blob/master/TrainingAnomalyDetector_public.py#L198
# before run install numpy==1.15.0 and theano==1.0.2

import theano.tensor as T
import theano


def paper_loss(y_true, y_pred):
    'Custom Objective function'
    # return T.sum(y_pred) + T.sum(y_true)
    y_true = T.flatten(y_true)
    y_pred = T.flatten(y_pred)

    n_seg = 32  # Because we have 32 segments per video.
    nvid = 60  # TODO: set batch size here
    n_exp = nvid // 2
    Num_d=32*nvid


    sub_max = T.ones_like(y_pred) # sub_max represents the highest scoring instants in bags (videos).
    sub_sum_labels = T.ones_like(y_true) # It is used to sum the labels in order to distinguish between normal and abnormal videos.
    sub_sum_l1=T.ones_like(y_true)  # For holding the concatenation of summation of scores in the bag.
    sub_l2 = T.ones_like(y_true) # For holding the concatenation of L2 of score in the bag.

    for ii in range(nvid):
        # For Labels
        mm = y_true[ii * n_seg:ii * n_seg + n_seg]
        sub_sum_labels = T.concatenate([sub_sum_labels, T.stack(T.sum(mm))])  # Just to keep track of abnormal and normal vidoes

        # For Features scores
        Feat_Score = y_pred[ii * n_seg:ii * n_seg + n_seg]
        sub_max = T.concatenate([sub_max, T.stack(T.max(Feat_Score))])         # Keep the maximum score of scores of all instances in a Bag (video)
        sub_sum_l1 = T.concatenate([sub_sum_l1, T.stack(T.sum(Feat_Score))])   # Keep the sum of scores of all instances in a Bag (video)

        z1 = T.ones_like(Feat_Score)
        z2 = T.concatenate([z1, Feat_Score])
        z3 = T.concatenate([Feat_Score, z1])
        z_22 = z2[31:]
        z_44 = z3[:33]
        z = z_22 - z_44
        z = z[1:32]
        z = T.sum(T.sqr(z))
        sub_l2 = T.concatenate([sub_l2, T.stack(z)])


    # sub_max[Num_d:] means include all elements after Num_d.
    # AllLabels =[2 , 4, 3 ,9 ,6 ,12,7 ,18 ,9 ,14]
    # z=x[4:]
    #[  6.  12.   7.  18.   9.  14.]

    sub_score = sub_max[Num_d:]  # We need this step since we have used T.ones_like
    F_labels = sub_sum_labels[Num_d:] # We need this step since we have used T.ones_like
    #  F_labels contains integer 32 for normal video and 0 for abnormal videos. This because of labeling done at the end of "load_dataset_Train_batch"



    # AllLabels =[2 , 4, 3 ,9 ,6 ,12,7 ,18 ,9 ,14]
    # z=x[:4]
    # [ 2 4 3 9]... This shows 0 to 3 elements

    sub_sum_l1 = sub_sum_l1[Num_d:] # We need this step since we have used T.ones_like
    sub_sum_l1 = sub_sum_l1[:n_exp]
    sub_l2 = sub_l2[Num_d:]         # We need this step since we have used T.ones_like
    sub_l2 = sub_l2[:n_exp]


    indx_nor = theano.tensor.eq(F_labels, 32).nonzero()[0]  # Index of normal videos: Since we labeled 1 for each of 32 segments of normal videos F_labels=32 for normal video
    indx_abn = theano.tensor.eq(F_labels, 0).nonzero()[0]

    n_Nor=n_exp

    Sub_Nor = sub_score[indx_nor] # Maximum Score for each of abnormal video
    Sub_Abn = sub_score[indx_abn] # Maximum Score for each of normal video

    z = T.ones_like(y_true)
    for ii in range(n_Nor):
        sub_z = T.maximum(1 - Sub_Abn + Sub_Nor[ii], 0)
        z = T.concatenate([z, T.stack(T.sum(sub_z))])

    z = z[Num_d:]  # We need this step since we have used T.ones_like
    z = T.mean(z, axis=-1) +  1.0*T.sum(sub_sum_l1) + 1.0*T.sum(sub_l2)  # Final Loss f

    return z
