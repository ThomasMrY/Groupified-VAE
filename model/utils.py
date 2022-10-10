import os
import torch
def save_checkpoint(net,optim,global_iter,filename, silent=True):
        model_states = {'net':net.state_dict(),}
        optim_states = {'optim':optim.state_dict(),}
        states = {'iter':global_iter,
              'model_states':model_states,
              'optim_states':optim_states}

        file_path = os.path.join("", filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path,global_iter))

def load_checkpoint(global_iter,net,optim,filename):
    file_path = os.path.join("", filename)
    if os.path.isfile(file_path):
        checkpoint = torch.load(file_path)
        global_iter = checkpoint['iter']
        net.load_state_dict(checkpoint['model_states']['net'])
        try:
            optim.load_state_dict(checkpoint['optim_states']['optim'])
        except:
            pass
        print("=> loaded checkpoint '{} (iter {})'".format(file_path,global_iter))
    else:
        print("=> no checkpoint found at '{}'".format(file_path))
    return global_iter,net,optim