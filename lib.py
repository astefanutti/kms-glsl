from ctypes import *
import argparse
import re

glsl = CDLL("./glsl.so")


class Metadata(argparse.Action):

    def __call__(self, parser, namespace, values, option=None):
        m = re.search(r'^(\w+)\.(\w+)$', values[0])
        if not m:
            raise ValueError(f'value {values[0]} for option {option} must match <UNIFORM>.KEY')
        metadata = getattr(namespace, self.dest)
        if not m.group(1) in metadata:
            metadata[m.group(1)] = {}
        metadata[m.group(1)] = {**metadata[m.group(1)], **{m.group(2): values[1]}}


class OPTIONS(Structure):
    _fields_ = [
        ("device",          c_char_p),
        ("mode",            c_ubyte * 32),
        ("format",          c_uint32),
        ("modifier",        c_uint64),
        ("async_page_flip", c_bool),
        ("atomic_drm_mode", c_bool),
        ("surfaceless",     c_bool),
        ("vrefresh",        c_int),
        ("count",           c_int),
    ]


def options(args):
    c_opts = OPTIONS()
    if args.async_page_flip:
        c_opts.async_page_flip = c_bool(True)
    if args.atomic_drm_mode:
        c_opts.atomic_drm_mode = c_bool(True)
    if args.device:
        c_opts.device = bytes(args.device.as_posix(), 'utf-8')
    if args.mode:
        c_opts.mode = (c_ubyte * 32)(*bytes(args.mode, 'utf-8'))
    return c_opts
