from ctypes import *

glsl = CDLL("./glsl.so")


class OPTIONS(Structure):
    _fields_ = [
        ("device",          c_char_p),
        ("mode",            c_ubyte * 32),
        ("format",          c_uint32),
        ("modifier",        c_uint64),
        ("connector",       c_int),
        ("async_page_flip", c_bool),
        ("atomic_drm_mode", c_bool),
        ("surfaceless",     c_bool),
        ("vrefresh",        c_int),
        ("frames",          c_uint),
    ]


def options(args):
    c_opts = OPTIONS()
    if args.async_page_flip:
        c_opts.async_page_flip = c_bool(True)
    if args.atomic_drm_mode:
        c_opts.atomic_drm_mode = c_bool(True)
    if args.connector:
        c_opts.connector = c_int(args.connector)
    else:
        c_opts.connector = -1
    if args.device:
        c_opts.device = bytes(args.device.as_posix(), 'utf-8')
    if args.mode:
        c_opts.mode = (c_ubyte * 32)(*bytes(args.mode, 'utf-8'))
    if args.frames:
        c_opts.frames = c_uint(args.frames)
    return c_opts
