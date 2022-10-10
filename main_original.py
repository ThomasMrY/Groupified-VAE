import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import tqdm as tqdm
import torch.optim as optimizer
from model import share
from torch.autograd import Variable
from model.Beta_VAE import BetaVAE_B,kl_divergence,reconstruction_loss
from model.Factor_VAE import FactorVAE1, Discriminator, permute_dims
from model.BetaTCVAE import BetaTCVAE, total_correlation
from model.utils import load_checkpoint, save_checkpoint
from datasets.data import return_data
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from visualize.visualize import Visualizer
from study.sweep import UnsupervisedStudyV1
import random
from data.ground_truth import named_data

import gin
import shutil
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument(r'--config_num', required=True, help=r'the number of settings of hyperparameters and random seeds')
parser.add_argument(r'--eval', action=r'store_true', help=r'eval model or not (default: False)')
args = parser.parse_args()

import types

def mkre_path(name):
    if name.lower() == 'dsprites_full':
        root_path = "./experiments/dsprites_full/"
    elif name.lower() == 'shapes3d':
        root_path = "./experiments/shapes3d/"
    elif name.lower() == "color_dsprites":
        root_path = "./experiments/color_dsprites/"
    elif name.lower() == "noisy_dsprites":
        root_path = "./experiments/noisy_dsprites/"
    elif name.lower() == "cars3d":
        root_path = "./experiments/cars3d/"
    elif name.lower() == "overlap144":
        root_path = "./experiments/overlap144/"
    elif name.lower() == "overlap1920":
        root_path = "./experiments/overlap1920/"
    else:
        raise NotImplementedError
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    return root_path


def file_remove(file_name):
    if os.path.exists(file_name) and os.path.isdir(file_name):
        shutil.rmtree(file_name)
    elif os.path.exists(file_name):
        os.remove(file_name)

def make_path(config_num):
    file_remove(root_path + config_num)
    os.mkdir(root_path + config_num)

@gin.configurable
def get_model(
    z_dim = 10,
    nc = gin.REQUIRED,
    N = 10,
    group = True,
    my_net = BetaVAE_B
    ):
    print([z_dim,group,my_net.__name__,nc])
    net = my_net(z_dim,nc,N,group).cuda()
    if group:
        net.complexfy = types.MethodType(share.complexfy, net)
        net.forward_action = types.MethodType(share.forward_action, net)
        net.action_order = types.MethodType(share.action_order_v2, net)
        net.abel_action = types.MethodType(share.abel_action, net)
        net.backward_action = types.MethodType(share.backward_action, net)
    return net


###################################              BetaVAE & AnnealVAE training code            ###################################
@gin.configurable
def train_BetaVAE(
    net,
    model_num,
    decoder_dist = "bernoulli",
    objective = 'B',
    lr = 1e-4,
    beta1 = 0.9,
    beta2 = 0.999,
    max_iter = 1e4*5,
    C_max = 25,
    C_stop_iter = 1e5,
    beta = 4,
    gamma = 10,
    weight = gin.REQUIRED,
    display_step = 200,
    save_step = 1000,
    visualize = False
    ):
    out = False
    model_name = root_path + str(model_num) + "/" + "model"
    net.model_name = model_name
    writer = SummaryWriter(model_name.replace("model",'vis'))
    global_iter = 0
    fst_iter = 2000
    optim = optimizer.Adam(net.parameters(), lr=lr,betas=(beta1, beta2))
    pbar = tqdm.tqdm(total=max_iter)
    pbar.update(global_iter)
    C_max = Variable(torch.FloatTensor([C_max]).cuda())
    while not out:
        for x in data_loader:
            global_iter += 1
            pbar.update(1)
            x = Variable(x.cuda())
            x_recon, mu, logvar = net(x)
            recon_loss = reconstruction_loss(x, x_recon, decoder_dist)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            net.global_iter = global_iter
                    

            if objective == 'H':
                if net.group:
                    if global_iter < fst_iter:
                        beta_vae_loss = recon_loss + beta*total_kld
                    elif global_iter == fst_iter:
                        mean_dims = share.check_dims(net,x)[0].tolist()   
                        beta_vae_loss = recon_loss + beta*total_kld
                    elif global_iter > fst_iter:
                        if global_iter % 200 == 0:
                            mean_dims = share.check_dims(net,x)[0].tolist()
                        if len(mean_dims) < 2:
                            beta_vae_loss = recon_loss + beta*total_kld
                        else:
                            gloss = share.group_constrains(net,x,mean_dims=mean_dims)
                            beta_vae_loss = recon_loss + beta*total_kld + weight*gloss
                else:
                    beta_vae_loss = recon_loss + beta*total_kld
            elif objective == 'B':
                C = torch.clamp(C_max/C_stop_iter*global_iter, 0, C_max.item())
                if net.group:
                    gloss = share.group_constrains(net,x)
                    beta_vae_loss = recon_loss + gamma*(total_kld-C).abs() + weight*gloss
                else:
                    beta_vae_loss = recon_loss + gamma*(total_kld-C).abs()

            optim.zero_grad()
            beta_vae_loss.backward()
            optim.step()
            if global_iter%display_step == 0:
                if net.group:
                    if objective == 'H' and global_iter > fst_iter and len(mean_dims) >= 2:
                        pbar.write('[{}] recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f} gloss:{:.3f}'.format(
                        global_iter, recon_loss.item(), total_kld.item(), mean_kld.item(),gloss.item()))
                        if visualize:
                            writer.add_scalar('loss/gloss', gloss.item(), global_iter)
                    elif objective == 'B':
                        pbar.write('[{}] recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f} gloss:{:.3f}'.format(
                        global_iter, recon_loss.item(), total_kld.item(), mean_kld.item(),gloss.item()))
                        if visualize:
                            writer.add_scalar('loss/gloss', gloss.item(), global_iter)
                    else:
                        pbar.write('[{}] recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f}'.format(
                        global_iter, recon_loss.item(), total_kld.item(), mean_kld.item()))
                else:
                    pbar.write('[{}] recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f}'.format(
                    global_iter, recon_loss.item(), total_kld.item(), mean_kld.item()))
                if visualize:
                    writer.add_scalar('loss/recon_loss', recon_loss.item(), global_iter)
                    writer.add_scalar('loss/total_kld', total_kld.item(), global_iter)
                    writer.add_scalars('loss/dim_wise_kld',{'dim_0':dim_wise_kld[0].item(),
                                                    'dim_1':dim_wise_kld[1].item(),
                                                    'dim_2':dim_wise_kld[2].item(),
                                                    'dim_3':dim_wise_kld[3].item(),
                                                    'dim_4':dim_wise_kld[4].item(),
                                                    'dim_5':dim_wise_kld[5].item(),
                                                    'dim_6':dim_wise_kld[6].item(),
                                                    'dim_7':dim_wise_kld[7].item(),
                                                    'dim_8':dim_wise_kld[8].item(),
                                                    'dim_9':dim_wise_kld[9].item()},global_iter)
                    dim_mu = mu.mean(0)
                    dim_var = logvar.exp().mean(0)
                    writer.add_scalars('mu_var/dim_mean',{'dim_0':dim_mu[0].item(),
                                                    'dim_1':dim_mu[1].item(),
                                                    'dim_2':dim_mu[2].item(),
                                                    'dim_3':dim_mu[3].item(),
                                                    'dim_4':dim_mu[4].item(),
                                                    'dim_5':dim_mu[5].item(),
                                                    'dim_6':dim_mu[6].item(),
                                                    'dim_7':dim_mu[7].item(),
                                                    'dim_8':dim_mu[8].item(),
                                                    'dim_9':dim_mu[9].item()},global_iter)
                    writer.add_scalars('mu_var/dim_var',{'dim_0':dim_var[0].item(),
                                                    'dim_1':dim_var[1].item(),
                                                    'dim_2':dim_var[2].item(),
                                                    'dim_3':dim_var[3].item(),
                                                    'dim_4':dim_var[4].item(),
                                                    'dim_5':dim_var[5].item(),
                                                    'dim_6':dim_var[6].item(),
                                                    'dim_7':dim_var[7].item(),
                                                    'dim_8':dim_var[8].item(),
                                                    'dim_9':dim_var[9].item()},global_iter)


                    writer.add_image("traversal", visl.visual_traversal(net),global_iter)
                    writer.flush()

            if global_iter%save_step == 0:
                save_checkpoint(net,optim,global_iter,model_name)
            if global_iter >= max_iter:
                out = True
                break
    return net

###################################                   BetaTCVAE training code                 ###################################
@gin.configurable
def train_BetaTCVAE2Stage(
    net,
    model_num,
    decoder_dist = "bernoulli",
    lr = 1e-4,
    beta1 = 0.9,
    beta2 = 0.999,
    max_iter = gin.REQUIRED,
    fst_iter = gin.REQUIRED,
    beta = gin.REQUIRED,
    weight = gin.REQUIRED,
    display_step = 200,
    save_step = 1000,
    visualize = False
    ):
    out = False
    model_name = root_path + str(model_num) + "/" + "model"
    net.model_name = model_name
    writer = SummaryWriter(model_name.replace("model",'vis'))
    global_iter = 0
    optim = optimizer.Adam(net.parameters(), lr=lr,betas=(beta1, beta2))
    pbar = tqdm.tqdm(total=max_iter)
    pbar.update(global_iter)
    while not out:
        for x in data_loader:
            global_iter += 1
            pbar.update(1)
            x = Variable(x.cuda())
            x_recon, mu, logvar,z = net(x)
            recon_loss = -reconstruction_loss(x, x_recon, decoder_dist)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            tc = total_correlation(net, beta, z, mu, logvar)

            if net.group:
                if global_iter < fst_iter:
                    vae_loss = (recon_loss + tc).mean().mul(-1)
                elif global_iter == fst_iter:
                    mean_dims = share.check_dims(net,x)[0].tolist()   
                    vae_loss = (recon_loss + tc).mean().mul(-1)
                elif global_iter > fst_iter:
                    if global_iter % 200 == 0:
                        mean_dims = share.check_dims(net,x)[0].tolist()
                    if len(mean_dims) < 2:
                        vae_loss = (recon_loss + tc).mean().mul(-1)
                    else:
                        gloss = share.group_constrains(net,x,mean_dims=mean_dims)
                        vae_loss = (recon_loss + tc).mean().mul(-1) + weight*gloss
            else:
                vae_loss = (recon_loss + tc).mean().mul(-1)
            

            optim.zero_grad()
            vae_loss.backward()
            optim.step()
            if global_iter%display_step == 0:
                if net.group and global_iter > fst_iter and len(mean_dims) >= 2:
                    pbar.write('[{}] recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f} gloss:{:.3f}'.format(
                    global_iter, recon_loss.item(), total_kld.item(), mean_kld.item(),gloss.item()))
                    if visualize:
                        writer.add_scalar('loss/gloss', gloss.item(), global_iter)
                else:
                    pbar.write('[{}] recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f}'.format(
                    global_iter, recon_loss.item(), total_kld.item(), mean_kld.item()))
                if visualize:
                    writer.add_scalar('loss/recon_loss', recon_loss.item(), global_iter)
                    writer.add_scalar('loss/total_kld', total_kld.item(), global_iter)
                    writer.add_scalar('loss/tc', tc.mean().item(), global_iter)
                    writer.add_scalars('loss/dim_wise_kld',{'dim_0':dim_wise_kld[0].item(),
                                                    'dim_1':dim_wise_kld[1].item(),
                                                    'dim_2':dim_wise_kld[2].item(),
                                                    'dim_3':dim_wise_kld[3].item(),
                                                    'dim_4':dim_wise_kld[4].item(),
                                                    'dim_5':dim_wise_kld[5].item(),
                                                    'dim_6':dim_wise_kld[6].item(),
                                                    'dim_7':dim_wise_kld[7].item(),
                                                    'dim_8':dim_wise_kld[8].item(),
                                                    'dim_9':dim_wise_kld[9].item()},global_iter)
                    dim_mu = mu.mean(0)
                    dim_var = logvar.exp().mean(0)
                    writer.add_scalars('mu_var/dim_mean',{'dim_0':dim_mu[0].item(),
                                                    'dim_1':dim_mu[1].item(),
                                                    'dim_2':dim_mu[2].item(),
                                                    'dim_3':dim_mu[3].item(),
                                                    'dim_4':dim_mu[4].item(),
                                                    'dim_5':dim_mu[5].item(),
                                                    'dim_6':dim_mu[6].item(),
                                                    'dim_7':dim_mu[7].item(),
                                                    'dim_8':dim_mu[8].item(),
                                                    'dim_9':dim_mu[9].item()},global_iter)
                    writer.add_scalars('mu_var/dim_var',{'dim_0':dim_var[0].item(),
                                                    'dim_1':dim_var[1].item(),
                                                    'dim_2':dim_var[2].item(),
                                                    'dim_3':dim_var[3].item(),
                                                    'dim_4':dim_var[4].item(),
                                                    'dim_5':dim_var[5].item(),
                                                    'dim_6':dim_var[6].item(),
                                                    'dim_7':dim_var[7].item(),
                                                    'dim_8':dim_var[8].item(),
                                                    'dim_9':dim_var[9].item()},global_iter)

                    writer.add_image("traversal", visl.visual_traversal(net),global_iter)
                    writer.flush()

            if global_iter%save_step == 0:
                save_checkpoint(net,optim,global_iter,model_name)
            if global_iter >= max_iter:
                out = True
                break
    return net

###################################                   FactorVAE training code                 ###################################
@gin.configurable
def train_factorVAE2Stage(net,
    model_num,
    decoder_dist = "bernoulli",
    lr = 1e-4,
    beta1 = 0.9,
    beta2 = 0.999,
    D_lr = 5e-5,
    D_beta1 = 0.9,
    D_beta2 = 0.999,
    gamma = gin.REQUIRED,
    weight = gin.REQUIRED,
    display_step = 200,
    save_step = 1000,
    max_iter = 1e4*8,
    fst_iter = 5e4,
    visualize = True
    ):
    out = False
    model_name = root_path + str(model_num) + "/" + "model"
    net.model_name = model_name
    writer = SummaryWriter(model_name.replace("model",'vis'))
    net.model_name = model_name
    global_iter = 0
    ones = torch.ones(32, dtype=torch.long).cuda()
    zeros = torch.zeros(32, dtype=torch.long).cuda()
    optim = optimizer.Adam(net.parameters(), lr=lr,betas=(beta1, beta2))
    D = Discriminator(net.z_dim).cuda()
    d_optim = optimizer.Adam(D.parameters(), lr=D_lr,betas=(D_beta1, D_beta2))
    pbar = tqdm.tqdm(total=max_iter)
    nets = [net, D]
    pbar.update(global_iter)
    while not out:
        for x1,x2 in data_loader:
            global_iter += 1
            pbar.update(1)
            x1 = Variable(x1.cuda())
            x_recon, mu, logvar, z = net(x1)
            recon_loss = reconstruction_loss(x1, x_recon, decoder_dist)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

            D_z = D(z)
            vae_tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()


            if net.group:
                if global_iter < fst_iter:
                    beta_vae_loss = recon_loss + total_kld + gamma*vae_tc_loss
                elif global_iter == fst_iter:
                    mean_dims = share.check_dims(net,x1)[0].tolist()   
                    beta_vae_loss = recon_loss + total_kld + gamma*vae_tc_loss
                else:
                    if global_iter % 200 == 0:
                        mean_dims = share.check_dims(net,x1)[0].tolist()
                    if len(mean_dims) < 2:
                        beta_vae_loss = recon_loss + total_kld  + gamma*vae_tc_loss
                    else:
                        gloss = share.group_constrains(net,x1,mean_dims=mean_dims)
                        beta_vae_loss = recon_loss + total_kld  + gamma*vae_tc_loss + weight*gloss
            else:
                beta_vae_loss = recon_loss + total_kld + gamma*vae_tc_loss


            optim.zero_grad()
            beta_vae_loss.backward(retain_graph=True)
            optim.step()

            x2 = Variable(x2.cuda())
            z_prime = net(x2, no_dec=True)
            z_pperm = permute_dims(z_prime).detach()
            
            D_z_pperm = D(z_pperm)
            D_z = D(z.detach())
            D_tc_loss = 0.5*(F.cross_entropy(D_z, zeros) + F.cross_entropy(D_z_pperm, ones))

            d_optim.zero_grad()
            D_tc_loss.backward()
            d_optim.step()


            if global_iter%display_step == 0:
                if net.group and global_iter > fst_iter and len(mean_dims) >= 2:
                    pbar.write('[{}] recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f} D_loss: {:.3f} gloss:{:.3f}'.format(
                    global_iter, recon_loss.item(), total_kld.item(), mean_kld.item(),D_tc_loss.item() ,gloss.item()))
                    if visualize:
                        writer.add_scalar('loss/gloss', gloss.item(), global_iter)
                else:
                    pbar.write('[{}] recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f} D_loss: {:.3f}'.format(
                    global_iter, recon_loss.item(), total_kld.item(), mean_kld.item(), D_tc_loss.item()))
                if visualize:
                    writer.add_scalar('loss/recon_loss', recon_loss.item(), global_iter)
                    writer.add_scalar('loss/total_kld', total_kld.item(), global_iter)
                    writer.add_scalar('loss/D_tc_loss', D_tc_loss.item(), global_iter)
                    writer.add_scalar('loss/vae_tc_loss', vae_tc_loss.item(), global_iter)
                    writer.add_scalars('loss/dim_wise_kld',{'dim_0':dim_wise_kld[0].item(),
                                                    'dim_1':dim_wise_kld[1].item(),
                                                    'dim_2':dim_wise_kld[2].item(),
                                                    'dim_3':dim_wise_kld[3].item(),
                                                    'dim_4':dim_wise_kld[4].item(),
                                                    'dim_5':dim_wise_kld[5].item(),
                                                    'dim_6':dim_wise_kld[6].item(),
                                                    'dim_7':dim_wise_kld[7].item(),
                                                    'dim_8':dim_wise_kld[8].item(),
                                                    'dim_9':dim_wise_kld[9].item()},global_iter)
                    dim_mu = mu.mean(0)
                    dim_var = logvar.exp().mean(0)
                    writer.add_scalars('mu_var/dim_mean',{'dim_0':dim_mu[0].item(),
                                                    'dim_1':dim_mu[1].item(),
                                                    'dim_2':dim_mu[2].item(),
                                                    'dim_3':dim_mu[3].item(),
                                                    'dim_4':dim_mu[4].item(),
                                                    'dim_5':dim_mu[5].item(),
                                                    'dim_6':dim_mu[6].item(),
                                                    'dim_7':dim_mu[7].item(),
                                                    'dim_8':dim_mu[8].item(),
                                                    'dim_9':dim_mu[9].item()},global_iter)
                    writer.add_scalars('mu_var/dim_var',{'dim_0':dim_var[0].item(),
                                                    'dim_1':dim_var[1].item(),
                                                    'dim_2':dim_var[2].item(),
                                                    'dim_3':dim_var[3].item(),
                                                    'dim_4':dim_var[4].item(),
                                                    'dim_5':dim_var[5].item(),
                                                    'dim_6':dim_var[6].item(),
                                                    'dim_7':dim_var[7].item(),
                                                    'dim_8':dim_var[8].item(),
                                                    'dim_9':dim_var[9].item()},global_iter)
                    
                    dim_Dz = D_z.mean(0)
                    dim_Dz_p = D_z_pperm.mean(0)
                    writer.add_scalars('D_output/z',{'dim_0':dim_Dz[0].item(),
                                                    'dim_1':dim_Dz[1].item()},global_iter)
                    writer.add_scalars('D_output/z_pperm',{'dim_0':dim_Dz_p[0].item(),
                                                    'dim_1':dim_Dz_p[1].item()},global_iter)

                    writer.add_image("traversal", visl.visual_traversal(net),global_iter)
                    writer.flush()

            if global_iter%save_step == 0:
                save_checkpoint(net,optim,global_iter,model_name)
            if global_iter%450000 == 0:
                save_checkpoint(net,optim,global_iter,model_name+"prime")
            if global_iter >= max_iter:
                out = True
                break
    return net

def write_text(metric,result_dict,print_txt,file):
    file = file.replace('model','eval')
    if os.path.exists(file):
        with open(file,'r') as f:
            new_dict = json.load(f)
    else:
        new_dict = {}
    new_dict[metric] = result_dict
    if print_txt:
        with open(file,'w') as f:
            json.dump(new_dict,f)

###################################                      Evaluation code                      ###################################
@gin.configurable
def evaluate(net,
    beta_VAE_score = False,
    dci_score = False,
    factor_VAE_score = False,
    MIG = False,
    Sap_score = False,
    unsupervised_score = False,
    print_txt = False
    ):
    dataset = visl.dataset
    def _representation(x):
        x = Variable(torch.from_numpy(x).float()).cuda()
        x = x.permute(0,3,1,2)
        z = net.encoder(x.contiguous())[:,:net.z_dim].squeeze()
        return z.detach().cpu().numpy()
    if beta_VAE_score:
        with gin.unlock_config():
            from evaluation.metrics.beta_vae import compute_beta_vae_sklearn
            gin.bind_parameter("beta_vae_sklearn.batch_size", 64)
            gin.bind_parameter("beta_vae_sklearn.num_train", 10000)
            gin.bind_parameter("beta_vae_sklearn.num_eval", 5000)
        result_dict = compute_beta_vae_sklearn(dataset,_representation,random_state=np.random.RandomState(0),artifact_dir=None)
        print("beta VAE score:" + str(result_dict))
        write_text("beta_VAE_score",result_dict,print_txt,net.model_name + ".json")
        gin.clear_config()
    if dci_score:
        from evaluation.metrics.dci import compute_dci
        with gin.unlock_config():
            gin.bind_parameter("dci.num_train", 10000)
            gin.bind_parameter("dci.num_test", 5000)
        result_dict = compute_dci(dataset,_representation,random_state=np.random.RandomState(0),artifact_dir=None)
        print("dci score:" + str(result_dict))
        write_text("dci_score",result_dict,print_txt,net.model_name + ".json")
        gin.clear_config()
    if factor_VAE_score:
        with gin.unlock_config():
            from evaluation.metrics.factor_vae import compute_factor_vae
            gin.bind_parameter("factor_vae_score.num_variance_estimate",10000)
            gin.bind_parameter("factor_vae_score.num_train",10000)
            gin.bind_parameter("factor_vae_score.num_eval",5000)
            gin.bind_parameter("factor_vae_score.batch_size",64)
            gin.bind_parameter("prune_dims.threshold",0.05)
        result_dict = compute_factor_vae(dataset,_representation,random_state=np.random.RandomState(0),artifact_dir=None)
        print("factor VAE score:" + str(result_dict))
        write_text("factor_VAE_score",result_dict,print_txt,net.model_name + ".json")
        gin.clear_config()
    if MIG:
        with gin.unlock_config():
            from evaluation.metrics.mig import compute_mig
            from evaluation.metrics.utils import _histogram_discretize
            gin.bind_parameter("mig.num_train",10000)
            gin.bind_parameter("discretizer.discretizer_fn",_histogram_discretize)
            gin.bind_parameter("discretizer.num_bins",20)
        result_dict = compute_mig(dataset,_representation,random_state=np.random.RandomState(0),artifact_dir=None)
        print("MIG score:" + str(result_dict))
        write_text("MIG",result_dict,print_txt,net.model_name + ".json")
        gin.clear_config()
    if unsupervised_score:
        with gin.unlock_config():
            from evaluation.metrics.unsupervised_metrics import unsupervised_metrics
            from evaluation.metrics.utils import _histogram_discretize
            gin.bind_parameter("unsupervised_metrics.num_train",10000)
            gin.bind_parameter("discretizer.discretizer_fn",_histogram_discretize)
            gin.bind_parameter("discretizer.num_bins",20)
        result_dict = unsupervised_metrics(dataset,_representation,random_state=np.random.RandomState(0),artifact_dir=None)
        print("unsupervised metrics score:" + str(result_dict))
        write_text("unsupervised_score",result_dict,print_txt,net.model_name + ".json")
        gin.clear_config()
    if Sap_score:
        with gin.unlock_config():
            from evaluation.metrics.sap_score import compute_sap
            gin.bind_parameter("sap_score.num_train",10000)
            gin.bind_parameter("sap_score.num_test",5000)
            gin.bind_parameter("sap_score.continuous_factors",False)
        result_dict = compute_sap(dataset,_representation,random_state=np.random.RandomState(0),artifact_dir=None)
        write_text("Sap_score",result_dict,print_txt,net.model_name + ".json")
        print("Sap score:" + str(result_dict))
        gin.clear_config()
###################################           Other metrics can be implemented here           ###################################


@gin.configurable
def main(
    model_num = 0,
    train_fc = gin.REQUIRED,
    name = gin.REQUIRED,
    eval = gin.REQUIRED
    ):
    net = get_model()
    train_fc(net=net,model_num = model_num)
    if eval:
        evaluate(net=net)

@gin.configurable
def main_eval(
    model_num = 0
    ):
    net = get_model()
    model_name = root_path + str(model_num) + "/" + "model"
    net.model_name = model_name
    global_iter = 0
    optim = None
    global_iter,net,optim = load_checkpoint(global_iter,net,optim,model_name)
    evaluate(net=net)

@gin.configurable
def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

my_study = UnsupervisedStudyV1()
model_bindings, model_config_file = my_study.get_model_config(int(args.config_num))
metric_config_file = my_study.get_eval_config_files()
gin.parse_config_files_and_bindings([model_config_file,metric_config_file], model_bindings)
visl = Visualizer()
root_path = mkre_path(visl.name)
if not args.eval:
    make_path(args.config_num)
my_study.print_model_config(int(args.config_num))
my_study.write_model_config(root_path + args.config_num,int(args.config_num))
random_seed()
data_loader = return_data(images = visl.dataset.images)
if not args.eval:
    main(int(args.config_num))
else:
    main_eval(int(args.config_num))