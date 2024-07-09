import os 
import numpy as np
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_npy")

key_mean_shape = np.load(f"{data_dir}/key_mean_shape.npy").reshape(1,-1)
id_base = np.load(f"{data_dir}/key_id_base.npy")
exp_base = np.load(f"{data_dir}/key_exp_base.npy")
lrs3_stats = np.load(f"{data_dir}/lrs3_stats.npy", allow_pickle=True).item()
lrs3_idexp_mean = lrs3_stats['idexp_lm3d_mean'].reshape([1, 68, 3])
lrs3_idexp_std = lrs3_stats['idexp_lm3d_std'].reshape([1, 68, 3])

def reconstruct_idexp_lm3d(id_coeff, exp_coeff, add_mean_face=False):
    """
    Generate 3D landmark with keypoint base!
    id_coeff: Tensor[T, c=80]
    exp_coeff: Tensor[T, c=64]
    """
    print("id_base : ", id_base.shape)
    print("exp_base : ", exp_base.shape)
    print("id_coeff : ", id_coeff.shape)
    print("exp_coeff : ", exp_coeff.shape)
    identity_diff_face = id_coeff@id_base.transpose(1,0) # [t,c],[c,3*68] ==> [t,3*68]
    expression_diff_face = exp_coeff @exp_base.transpose(1,0) # [t,c],[c,3*68] ==> [t,3*68]
    
    face = identity_diff_face + expression_diff_face # [t,3N]
    face = face.reshape([face.shape[0], -1, 3]) # [t,N,3]
    if add_mean_face:
        lm3d = face + key_mean_shape.reshape([1, 68, 3]) # [3*68, 1] ==> [1, 3*68]
    else:
        lm3d = face*10
    return lm3d

def denorm_lm3d_to_lm2d(lm3d, face_size=512):
    """
        lm2d = (lm2d + 1.0) / 2.0 \n
        lm2d[:, :, 1] = 1.0 - lm2d[:, :, 1]
    """
    lm2d = lm3d[:, :, :2]
    lm2d = (lm2d + 1.0) / 2.0
    lm2d[:, :, 1] = 1.0 - lm2d[:, :, 1]
    lm2d *= face_size
    return lm2d
