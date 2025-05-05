from modules import LMHTNeuron, LMHT_Inference_Neuron, IFNeuron, QCFS
from spikingjelly.clock_driven import layer

def isActivation(name):
    if 'relu' in name.lower() or 'qcfs' in name.lower():
        return True
    return False

def isLayer(name):
    if 'sync' in name.lower() or '2d' in name.lower() or 'linear' in name.lower() or 'dropout' in name.lower():
        return True
    return False


def replace_activation_by_LMHT(model, L, T, init_mem=0.):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_activation_by_LMHT(module, L, T, init_mem)
        if isActivation(module.__class__.__name__.lower()):
            if L > 1:
                model._modules[name] = LMHTNeuron(L, T, 2./L, init_mem)
            else:
                model._modules[name] = IFNeuron(T, 1., init_mem)
    return model


def replace_by_LMHT_Inference(model, L, T, init_mem=0.):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_by_LMHT_Inference(module, L, T, init_mem)
        if module.__class__.__name__.lower() == "lmhtneuron":
            model._modules[name] = LMHT_Inference_Neuron(L, module.alpha, module.mask, module.mask_linear, L*T, module.v_threshold.item(), init_mem)
            
    return model

    
    
def replace_layer_bias(model, L):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_layer_bias(module, L)
        if isLayer(module.__class__.__name__.lower()) and hasattr(module, "bias"):
            try:
                module.bias.requires_grad = False
                module.bias /= L
                if module.__class__.__name__.lower() == 'batchnorm2d' or 'sync' in module.__class__.__name__.lower():
                    module.running_mean /= L
            except:
                pass
    return model  


def replace_activation_by_QCFS(model, T, thresh):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_activation_by_QCFS(module, T, thresh)
        if isActivation(module.__class__.__name__.lower()):
            model._modules[name] = QCFS(up=thresh, t=T)
    return model


def replace_layer_by_snn_layer(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_layer_by_snn_layer(module)
        if isLayer(module.__class__.__name__.lower()):
            model._modules[name] = layer.SeqToANNContainer(module)
    return model    


def replace_QCFS_by_IFNode(model, T):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_QCFS_by_IFNode(module, T)
        if isActivation(module.__class__.__name__.lower()):
            model._modules[name] = IFNeuron(T=T, th=module.thresh.item(), inital_mem=0.5)
    return model


def replace_IFNode_by_LMHT(model, L, T):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_IFNode_by_LMHT(module, L, T)
        if module.__class__.__name__.lower() == "ifneuron":
            model._modules[name] = LMHTNeuron(L, T, module.v_threshold.item(), 0.5)
    return model


def error(info):
    print(info)
    exit(1)
