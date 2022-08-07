import io
import os
import os.path as osp
import re
import time
import warnings
from collections import OrderedDict
from importlib import import_module
from tempfile import TemporaryDirectory
import torch
import torchvision
from torch.optim import Optimizer
from alfred.dl.torch.distribute.utils import get_dist_info
from torch.utils.model_zoo import load_url
import os
import shutil
import torch
import os.path as osp
from alfred import logger as LOGGER
from torch import nn
from alfred import logger


def is_module_wrapper(module):
    return False


class CheckpointLoader:
    """A general checkpoint loader to manage all schemes."""

    _schemes = {}

    @classmethod
    def _register_scheme(cls, prefixes, loader, force=False):
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        else:
            assert isinstance(prefixes, (list, tuple))
        for prefix in prefixes:
            if (prefix not in cls._schemes) or force:
                cls._schemes[prefix] = loader
            else:
                raise KeyError(
                    f"{prefix} is already registered as a loader backend, "
                    'add "force=True" if you want to override it'
                )
        # sort, longer prefixes take priority
        cls._schemes = OrderedDict(
            sorted(cls._schemes.items(), key=lambda t: t[0], reverse=True)
        )

    @classmethod
    def register_scheme(cls, prefixes, loader=None, force=False):
        """Register a loader to CheckpointLoader.

        This method can be used as a normal class method or a decorator.

        Args:
            prefixes (str or list[str] or tuple[str]):
            The prefix of the registered loader.
            loader (function, optional): The loader function to be registered.
                When this method is used as a decorator, loader is None.
                Defaults to None.
            force (bool, optional): Whether to override the loader
                if the prefix has already been registered. Defaults to False.
        """

        if loader is not None:
            cls._register_scheme(prefixes, loader, force=force)
            return

        def _register(loader_cls):
            cls._register_scheme(prefixes, loader_cls, force=force)
            return loader_cls

        return _register

    @classmethod
    def _get_checkpoint_loader(cls, path):
        """Finds a loader that supports the given path. Falls back to the local
        loader if no other loader is found.

        Args:
            path (str): checkpoint path

        Returns:
            callable: checkpoint loader
        """
        for p in cls._schemes:
            # use regular match to handle some cases that where the prefix of
            # loader has a prefix. For example, both 's3://path' and
            # 'open-mmlab:s3://path' should return `load_from_ceph`
            if re.match(p, path) is not None:
                return cls._schemes[p]

    @classmethod
    def load_checkpoint(cls, filename, map_location=None):
        """load checkpoint through URL scheme path.

        Args:
            filename (str): checkpoint file name with given prefix
            map_location (str, optional): Same as :func:`torch.load`.
                Default: None
            logger (:mod:`logging.Logger`, optional): The logger for message.
                Default: None

        Returns:
            dict or OrderedDict: The loaded checkpoint.
        """

        checkpoint_loader = cls._get_checkpoint_loader(filename)
        class_name = checkpoint_loader.__name__
        logger.info(f"load checkpoint from {class_name[10:]} path: {filename}")
        return checkpoint_loader(filename, map_location)


@CheckpointLoader.register_scheme(prefixes="")
def load_from_local(filename, map_location):
    """load checkpoint by local file path.

    Args:
        filename (str): local checkpoint file path
        map_location (str, optional): Same as :func:`torch.load`.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    filename = osp.expanduser(filename)
    if not osp.isfile(filename):
        raise FileNotFoundError(f"{filename} can not be found.")
    checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint


@CheckpointLoader.register_scheme(prefixes=("http://", "https://"))
def load_from_http(filename, map_location=None, model_dir=None):
    """load checkpoint through HTTP or HTTPS scheme path. In distributed
    setting, this function only download checkpoint at local rank 0.

    Args:
        filename (str): checkpoint file path with modelzoo or
            torchvision prefix
        map_location (str, optional): Same as :func:`torch.load`.
        model_dir (string, optional): directory in which to save the object,
            Default: None

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    # print('loading model from http.')
    rank, world_size = get_dist_info()
    if rank == 0:
        checkpoint = load_url(filename, model_dir=model_dir, map_location=map_location)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            checkpoint = load_url(
                filename, model_dir=model_dir, map_location=map_location
            )
    return checkpoint


def _load_checkpoint(filename, map_location=None):
    return CheckpointLoader.load_checkpoint(filename, map_location)


def load_state_dict(module, state_dict, strict=False, logger=None):
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=""):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        if is_module_wrapper(module):
            module = module.module
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            all_missing_keys,
            unexpected_keys,
            err_msg,
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(module)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [key for key in all_missing_keys if "num_batches_tracked" not in key]

    if unexpected_keys:
        err_msg.append(
            "unexpected key in source " f'state_dict: {", ".join(unexpected_keys)}\n'
        )
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n'
        )

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(0, "The model and loaded state dict do not match exactly\n")
        err_msg = "\n".join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


def load_checkpoint(
    model,
    filename,
    map_location=None,
    strict=False,
    revise_keys=[(r"^module\.", "")],
    force_return=False,
):
    logger.info(f"loading weights from: {filename}, will automaticall choose schema..")
    checkpoint = _load_checkpoint(filename, map_location)
    logger.info("Loaded checkpoint.")
    if force_return:
        return checkpoint
    # print(checkpoint)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"No state_dict found in checkpoint file {filename}")
    # get state_dict from checkpoint
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # strip prefix of state_dict
    metadata = getattr(state_dict, "_metadata", OrderedDict())
    for p, r in revise_keys:
        state_dict = OrderedDict({re.sub(p, r, k): v for k, v in state_dict.items()})
    # Keep metadata in state_dict
    state_dict._metadata = metadata
    # load state_dict
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint


# Below for ....


def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = (
        torch.zeros(conv.weight.size(0), device=conv.weight.device)
        if conv.bias is None
        else conv.bias
    )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
        torch.sqrt(bn.running_var + bn.eps)
    )
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def fuse_model(model):
    # from yolov6.layers.common import Conv
    # for m in model.modules():
    #     if type(m) is Conv and hasattr(m, "bn"):
    #         m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
    #         delattr(m, "bn")  # remove batchnorm
    #         m.forward = m.forward_fuse  # update forward
    return model


def load_state_dict_v6(weights, model, map_location=None):
    """Load weights from checkpoint file, only assign weights those layers' name and shape are match."""
    ckpt = torch.load(weights, map_location=map_location)
    state_dict = ckpt["model"].float().state_dict()
    model_state_dict = model.state_dict()
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if k in model_state_dict and v.shape == model_state_dict[k].shape
    }
    model.load_state_dict(state_dict, strict=False)
    del ckpt, state_dict, model_state_dict
    return model


def load_checkpoint_v6(weights, map_location=None, inplace=True, fuse=True):
    """Load model from checkpoint file."""
    LOGGER.info("Loading checkpoint from {}".format(weights))
    ckpt = torch.load(weights, map_location=map_location)  # load
    model = ckpt["ema" if ckpt.get("ema") else "model"].float()
    if fuse:
        LOGGER.info("\nFusing model...")
        model = fuse_model(model).eval()
    else:
        model = model.eval()
    return model


def save_checkpoint(ckpt, is_best, save_dir, model_name=""):
    """Save checkpoint to the disk."""
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    filename = osp.join(save_dir, model_name + ".pt")
    torch.save(ckpt, filename)
    if is_best:
        best_filename = osp.join(save_dir, "best_ckpt.pt")
        shutil.copyfile(filename, best_filename)


def strip_optimizer(ckpt_dir):
    for s in ["best", "last"]:
        ckpt_path = osp.join(ckpt_dir, "{}_ckpt.pt".format(s))
        if not osp.exists(ckpt_path):
            continue
        ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
        if ckpt.get("ema"):
            ckpt["model"] = ckpt["ema"]  # replace model with ema
        for k in ["optimizer", "ema", "updates"]:  # keys
            ckpt[k] = None
        ckpt["epoch"] = -1
        ckpt["model"].half()  # to FP16
        for p in ckpt["model"].parameters():
            p.requires_grad = False
        torch.save(ckpt, ckpt_path)
