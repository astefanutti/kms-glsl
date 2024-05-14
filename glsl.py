#!/usr/bin/env python

import argparse
import glob
import os
import signal
import stat
import threading

from contextlib import ExitStack
from lib import Metadata, options
from input import *
from libevdev import *
from signal import pthread_sigmask, pthread_kill, sigwait
from threading import main_thread

'''
"""
Example #1:
- Given the following uniform is declared in the shader:
  uniform int variable;
- Extend the Input class, and set the value of the uniform
  in the render method.
- Create an instance with the uniform variable name. 
"""


class Variable(Input):
    value = 0

    def init(self, program, width, height):
        """
        This method is called once after the program has been set up,
        before the rendering starts. It can be used to further initialise
        the input, using the `glsl` variable, that exposes OpenGL ES APIs.

        Args:
          program:
            The handle to the current program object.
          width:
            The width of the viewport.
          height:
            The height of the viewport.
        """
        super().init(program, width, height)

    def render(self, frame, time):
        """
        This method is called for every frame. It can be used to update
        the rendering state, using the `glsl` variable, that exposes OpenGL ES APIs.

        Args:
          frame:
            The current frame index, or frames since start time.
          time:
            The elapsed time since start time, in seconds.
        """
        self.value += 1
        glsl.glUniform1ui(self.loc, self.value)


Variable('variable')
'''

'''
"""
Example #2:
- Given the following uniforms are declared in the shader:
  uniform int iAux0;
  uniform vec4 iAux1;
- Declare an init callback function to retrieve the location
  of the uniform variables.
- Declare a render callback function to set the uniform values
  before each frame is rendered.
- Register the two callback functions.
"""

iAux0 = iAux1 = None
vAux0 = 0
vAux1 = [0] * 4


@CFUNCTYPE(None, c_uint, c_uint, c_uint)
def init(program, width, height):
    """
    This function is called once after the program has been set up,
    before the rendering starts. It can be used to further initialise
    the program, using the `glsl` variable, that exposes OpenGL ES APIs.

    Args:
      program:
        The handle to the current program object.
      width:
        The width of the viewport.
      height:
        The height of the viewport.
    """
    global iAux0, iAux1
    iAux0 = glsl.glGetUniformLocation(program, b'iAux0')
    iAux1 = glsl.glGetUniformLocation(program, b'iAux1')


@CFUNCTYPE(None, c_uint64, c_float)
def render(frame, time):
    """
    This function is called for every frame. It can be used to update
    the rendering state, using the `glsl` variable, that exposes OpenGL ES APIs.

    Args:
      frame:
        The current frame index, or frames since start time.
      time:
        The elapsed time since start time, in seconds.
    """
    global vAux0, vAux1
    vAux0 += 1
    glsl.glUniform1ui(iAux0, vAux0)
    vAux1 = [vAux0] * 4
    glsl.glUniform4f(iAux1, vAux1[0], vAux1[1], vAux1[2], vAux1[3])


glsl.onInit(init)
glsl.onRender(render)
'''

parser = argparse.ArgumentParser(description='Run OpenGL shaders using DRM/KMS')
parser.add_argument('shader', metavar='FILE', type=Path, nargs=1,
                    help='the shader file')
parser.add_argument('--async-page-flip', action=argparse.BooleanOptionalAction,
                    help='use async page flipping')
parser.add_argument('--atomic-drm-mode', action=argparse.BooleanOptionalAction,
                    help='use atomic mode setting and fencing')
parser.add_argument('-C', '--connector', metavar='CONNECTOR', type=int,
                    help='the DRM connector')
parser.add_argument('-D', '--device', metavar='DEVICE', type=Path,
                    help='the DRM device')
parser.add_argument('--mode', metavar='MODE', type=str,
                    help='specify the video mode in the format <resolution>[-<vrefresh>]')
parser.add_argument('-k', '--keyboard', metavar='UNIFORM', type=str,
                    help='add keyboard')
parser.add_argument('--touch', metavar='UNIFORM', type=str,
                    help='add touch device')
parser.add_argument('-c', '--cubemap', metavar=('UNIFORM', 'FILE'), type=str, nargs=2,
                    action='append', dest='cubemaps', default=[], help='add cubemap')
parser.add_argument('-t', '--texture', metavar=('UNIFORM', 'FILE'), type=str, nargs=2,
                    action='append', dest='textures', default=[], help='add texture')
parser.add_argument('-v', '--volume', metavar=('UNIFORM', 'FILE'), type=str, nargs=2,
                    action='append', dest='volumes', default=[], help='add volume')
parser.add_argument('-m', '--metadata', metavar=('<UNIFORM>.KEY', 'VALUE'), type=str, nargs=2,
                    action=Metadata, dest='metadata', default={}, help='set uniform metadata')
args = parser.parse_args()


def is_device(input_dev):
    if not os.path.exists(input_dev):
        return False

    m = os.stat(input_dev)[stat.ST_MODE]
    if not stat.S_ISCHR(m):
        return False

    return True


close_devices = None
with ExitStack() as stack:
    for path in list(filter(is_device, glob.glob('{}/event*'.format('/dev/input')))):
        dev = Device(stack.enter_context(open(path, 'rb')))
        if dev.has(EV_REL) and dev.has(EV_KEY.BTN_LEFT):
            # Mouse
            ButtonMouse('iMouse', dev)
        elif args.keyboard and dev.has(EV_KEY) and dev.has(EV_KEY.KEY_A):
            # Keyboard
            Keyboard(args.keyboard, dev)
        elif dev.has(EV_ABS) and dev.has(EV_KEY.BTN_TOUCH) and dev.has_property(INPUT_PROP_DIRECT):
            # Touchscreen
            # Only consider direct input devices, like touchscreens and drawing tablets, see:
            # https://www.kernel.org/doc/Documentation/input/event-codes.txt
            Touchscreen(args.touch if args.touch else 'iTouch', dev)
        else:
            dev.fd.close()
            continue
        dev.grab()

    close_devices = stack.pop_all().close

if args.keyboard and len(list(filter(lambda i: isinstance(i, Keyboard), inputs))) == 0:
    print(f'no keyboard device found for uniform {args.keyboard}')

if args.touch and len(list(filter(lambda i: isinstance(i, Touchscreen), inputs))) == 0:
    print(f'no touch device found for uniform {args.touch}')

for cubemap in args.cubemaps:
    CubemapTexture(cubemap[0], cubemap[1])
for texture in args.textures:
    ImageTexture(texture[0], texture[1], **args.metadata[texture[0]] if texture[0] in args.metadata else {})
for volume in args.volumes:
    VolumeTexture(volume[0], volume[1])

ret = glsl.init(bytes(args.shader[0].as_posix(), 'utf-8'), byref(options(args)))
if ret != 0:
    close_devices()
    exit(ret)
ret = glsl.run()
if ret != 0:
    close_devices()
    exit(ret)


stopped = threading.Event()
pthread_sigmask(signal.SIG_BLOCK, [signal.SIGCONT])


def join():
    glsl.join()
    stopped.set()
    pthread_kill(main_thread().ident, signal.SIGCONT)


Thread(target=join).start()

if sigwait({signal.SIGINT, signal.SIGCONT}) == signal.SIGINT:
    glsl.stop()
    stopped.wait(timeout=30)

close_devices()
