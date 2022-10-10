"""Abstract base class for a study."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Study(object):
  """Abstract base class used for different studies."""

  def get_model_config(self, model_num=0):
    """Returns model bindings and config file."""
    raise NotImplementedError()

  def print_model_config(self, model_num=0):
    """Prints model bindings and config file."""
    model_bindings, model_config_file = self.get_model_config(model_num)
    print("Gin base config for model training:")
    print("--")
    print(model_config_file)
    print()
    print("Gin bindings for model training:")
    print("--")
    for binding in model_bindings:
      print(binding)

  def get_postprocess_config_files(self):
    """Returns postprocessing config files."""
    raise NotImplementedError()

  def print_postprocess_config(self):
    """Prints postprocessing config files."""
    print("Gin config files for postprocessing (random seeds may be set "
          "later):")
    print("--")
    for path in self.get_postprocess_config_files():
      print(path)

  def get_eval_config_files(self):
    """Returns evaluation config files."""
    raise NotImplementedError()

  def print_eval_config(self):
    """Prints evaluation config files."""
    print("Gin config files for evaluation (random seeds may be set later):")
    print("--")
    for path in self.get_eval_config_files():
      print(path)
    
  def write_model_config(self, model_num=0):
    """Returns evaluation config files."""
    raise NotImplementedError()