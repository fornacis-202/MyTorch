from mytorch import Tensor

def CategoricalCrossEntropy(preds: Tensor, label: Tensor):
    "TODO: implement Categorical Cross Entropy loss"
    ce = label.__mul__(preds.log())
    # print("loss", ce.data ," ens\n")
    ce = ce.sum()
    ce = ce.__neg__()

    
    return ce
