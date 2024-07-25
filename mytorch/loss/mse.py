from mytorch import Tensor

def MeanSquaredError(preds: Tensor, actual: Tensor):
    "TODO: implement Mean Squared Error loss"
    subs_sum = preds.__sub__(actual)
    subs_sum = subs_sum.__pow__(2)
    subs_sum = subs_sum.sum()
    subs_sum = subs_sum.__mul__(1/preds.shape[0])
    # subs_sum.data /= preds.shape[0]
    return subs_sum

