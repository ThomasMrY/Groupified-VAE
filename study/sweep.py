from study import study
from utils import resources
import utils.hyperparams as h
from six.moves import range
import json


def get_datasets():
    """Returns all the data sets."""
    dataset_list = [
          "dsprites_full",
          "shapes3d",
          "color_dsprites",
          "noisy_dsprites",
          "cars3d",
          "overlap144",
          "overlap1920"
      ]
    return_data =  h.sweep(
      "return_data.name",
      h.categorical(dataset_list))
    visa_data = h.sweep(
      "Visualizer.name",
      h.categorical(dataset_list))
    nc = h.sweep(
      "get_model.nc",
      h.categorical([
          1, 3, 3,3,3, 1, 1
    ]))
    return h.zipit([return_data,visa_data, nc])

def get_num_latent(sweep):
  return h.sweep("encoder.num_latent", h.discrete(sweep))


def get_seeds(num):
  """Returns random seeds."""
  return h.sweep("random_seed.seed", h.categorical(list(range(num))))


def get_default_models():
    """Our default set of models (6 model * 6 hyperparameters=36 models)."""
    # BetaVAE config.
    model_name = h.fixed("main.name", "BetaVAE")
    betas = h.sweep("train_BetaVAE.beta", h.discrete([10., 20., 30.]))
    groupfy = h.sweep("get_model.group",h.discrete([False, True]))
    config_beta_vae = h.product([model_name, betas, groupfy])

    # AnnealedVAE config.
    model_name = h.fixed("main.name", "AnnealVAE")
    c = h.sweep("train_BetaVAE.C_max", h.discrete([10., 20., 30.]))
    max_iter = h.sweep("train_BetaVAE.max_iter", h.discrete([3e4, 4e4, 5e4]))
    C_stop_iter = h.sweep("train_BetaVAE.C_stop_iter", h.discrete([2e4, 3e4, 4e4]))
    iters = h.zipit([max_iter,C_stop_iter])
    groupfy = h.sweep("get_model.group",h.discrete([False, True]))
    config_annealed_beta_vae = h.product(
      [model_name, c, iters, groupfy])

    # FactorVAE config.
    model_name = h.fixed("main.name", "FactorVAE")
    gammas = h.sweep("train_factorVAE2Stage.gamma",
                   h.discrete([5., 10., 15.]))
    groupfy = h.sweep("get_model.group",h.discrete([False, True]))
    config_factor_vae = h.product([model_name, gammas, groupfy])

    # BetaTCVAE config.
    model_name = h.fixed("main.name", "BetaTCVAE")
    betas = h.sweep("train_BetaTCVAE2Stage.beta", h.discrete([6., 9., 12.]))
    groupfy = h.sweep("get_model.group",h.discrete([False, True]))
    config_beta_tc_vae = h.product([model_name, betas, groupfy])
    all_models = h.chainit([
      config_beta_vae, config_annealed_beta_vae, config_factor_vae,
      config_beta_tc_vae
    ])
    return all_models

def get_config():
    return h.product([
      get_datasets(),
      get_default_models(),
      get_seeds(10)
    ])

class UnsupervisedStudyV1(study.Study):
  """Defines the study for the paper."""

  def get_model_config(self, model_num=0):
    """Returns model bindings and config file."""
    config = get_config()[model_num]
    model_bindings = h.to_bindings(config)
    model_config_file = resources.get_file(
        "gins/" + config['main.name'] + ".gin")
    return model_bindings, model_config_file
  
  def write_model_config(self,path, model_num=0):
    config = get_config()[model_num]
    with open(path + "/config.json",'w+') as f:
        json.dump(config,f)
    

  
  def get_eval_config_files(self):
    return resources.get_file(
        "gins/" + "metric.gin")