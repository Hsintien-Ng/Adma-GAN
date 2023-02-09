import pickle 
import numpy as np
import torch
import torch.nn.functional as F


def l2_normalize(x, dim=None, epsilon=1e-12):
    square_sum = torch.sum(torch.square(x), dim=dim, keepdims=True)
    x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.ones_like(square_sum) * epsilon))
    return torch.multiply(x, x_inv_norm)


def get_statistics(logits, labels):
    """Gets accuracy and entropy."""
    prob = F.softmax(logits)
    entropy = - torch.mean(torch.sum(prob * torch.log(prob + 1e-8), axis=-1))
    label_acc = (torch.argmax(logits, dim=-1) == torch.argmax(labels, dim=-1))
    label_acc = torch.mean(label_acc.float())
    return label_acc, entropy


def load_adj(num_classes, t, path):
    assert path is not None
    with open(path, 'rb') as f:
        result = pickle.load(f)
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int)
    return _adj
    
    
def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


def contrastive_loss(image_feat, cond_feat, l2_norm=True, temperature=0.1, device="cpu"):
    """Calculates contrastive loss."""
    if l2_norm:
        image_feat = l2_normalize(image_feat, -1)
        cond_feat = l2_normalize(cond_feat, -1)
    local_batch_size = image_feat.shape[0]
    image_feat_large = image_feat
    cond_feat_large = cond_feat
    labels = torch.arange(local_batch_size).to(device)
    # labels_onehot = F.one_hot(labels, local_batch_size)
    logits_img2cond = torch.matmul(image_feat,
                                 cond_feat_large.t()) / temperature
    logits_cond2img = torch.matmul(cond_feat,
                                 image_feat_large.t()) / temperature
    loss_img2cond = F.cross_entropy(logits_img2cond, labels)
    loss_cond2img = F.cross_entropy(logits_cond2img, labels)
    loss_img2cond = torch.mean(loss_img2cond)
    loss_cond2img = torch.mean(loss_cond2img)
    loss = loss_img2cond + loss_cond2img
    # accuracy1, entropy1 = get_statistics(logits_img2cond, labels_onehot)
    # accuracy2, entropy2 = get_statistics(logits_cond2img, labels_onehot)
    # accuracy = 0.5 * (accuracy1 + accuracy2)
    # entropy = 0.5 * (entropy1 + entropy2)
    return loss


if __name__ == "__main__":
    image_feat = torch.rand(8, 256).cuda()
    cond_feat = torch.rand(8, 256).cuda()
    loss, acc, entropy = contrastive_loss(image_feat, cond_feat, device="cuda")
    print(loss, acc, entropy)