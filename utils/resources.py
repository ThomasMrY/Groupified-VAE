# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility to access resources in package."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os


def get_file(path):
  """Returns path relative to file."""
  return path


def get_files_in_folder(path):
  return [os.path.join(path,x) for x in os.listdir(path)]


