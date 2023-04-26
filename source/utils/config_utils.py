"""
 Copyright 2022 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import os
import yaml
from easydict import EasyDict as edict


# torch.backends.cudnn.enabled = False
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

def parse_arguments(args):
    """
    Parse arguments from command line.
    Syntax: --key1.key2.key3=value --> value
            --key1.key2.key3=      --> None
            --key1.key2.key3       --> True
            --key1.key2.key3!      --> False
    """
    opt_cmd = {}
    for arg in args:
        assert(arg.startswith("--"))
        if "=" not in arg[2:]:
            key_str,value = (arg[2:-1],"false") if arg[-1]=="!" else (arg[2:],"true")
        else:
            key_str,value = arg[2:].split("=")
        keys_sub = key_str.split(".")
        opt_sub = opt_cmd
        for k in keys_sub[:-1]:
            if k not in opt_sub: opt_sub[k] = {}
            opt_sub = opt_sub[k]
        assert keys_sub[-1] not in opt_sub,keys_sub[-1]
        opt_sub[keys_sub[-1]] = yaml.safe_load(value)
    opt_cmd = edict(opt_cmd)
    return opt_cmd

def override_options(opt,opt_over,key_stack=[],safe_check=False):
    """Overrides edict with a new edict. """
    for key,value in opt_over.items():
        if isinstance(value,dict):
            # parse child options (until leaf nodes are reached)
            opt[key] = override_options(opt.get(key,dict()),value,key_stack=key_stack+[key],safe_check=safe_check)
        else:
            # ensure command line argument to override is also in yaml file
            if safe_check and key not in opt:
                add_new = None
                while add_new not in ["y","n"]:
                    key_str = ".".join(key_stack+[key])
                    add_new = input("\"{}\" not found in original opt, add? (y/n) ".format(key_str))
                if add_new=="n":
                    print("safe exiting...")
                    exit()
            opt[key] = value
    return opt

def load_options(fname):
    """Load from yaml file and create edict. """
    with open(fname) as file:
        opt = edict(yaml.safe_load(file))
    if "_parent_" in opt:
        # load parent yaml file(s) as base options
        parent_fnames = opt.pop("_parent_")
        if type(parent_fnames) is str:
            parent_fnames = [parent_fnames]
        for parent_fname in parent_fnames:
            opt_parent = load_options(parent_fname)
            opt_parent = override_options(opt_parent,opt,key_stack=[])
            opt = opt_parent
    print("loading {}...".format(fname))
    return opt

def save_options_file(opt, save_dir, override=None):
    opt_fname = "{}/options.yaml".format(save_dir)
    if os.path.isfile(opt_fname):
        with open(opt_fname) as file:
            opt_old = yaml.safe_load(file)
        if opt!=opt_old:
            # prompt if options are not identical
            opt_new_fname = "{}/options_temp.yaml".format(save_dir)
            with open(opt_new_fname,"w") as file:
                yaml.safe_dump(to_dict(opt),file,default_flow_style=False,indent=4)
            print("existing options file found (different from current one)...")
            os.system("diff {} {}".format(opt_fname,opt_new_fname))
            os.system("rm {}".format(opt_new_fname))
            while override not in ["y","n"]:
                override = input("override? (y/n) ")
            if override=="n":
                print("safe exiting...")
                exit()
        else: print("existing options file found (identical)")
    else: print("(creating new options file...)")
    with open(opt_fname,"w") as file:
        yaml.safe_dump(to_dict(opt),file,default_flow_style=False,indent=4)
    return 

def to_dict(D, dict_type=dict):
    D = dict_type(D)
    for k,v in D.items():
        if isinstance(v,dict):
            D[k] = to_dict(v,dict_type)
    return D

def dict_compare(d1, d2):
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    shared_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = {o : (d1[o], d2[o]) for o in shared_keys if d1[o] != d2[o]}
    same = set(o for o in shared_keys if d1[o] == d2[o])
    return added, removed, modified, same