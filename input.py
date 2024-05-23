from ctypes import *
from typing import Generic, Optional, TypeVar

import collections
import signal
import threading

from errno import ENODEV
from gl import *
from libevdev import EV_ABS, EV_KEY, EV_REL, EventsDroppedException
from pathlib import Path
from PIL import Image
from lib import glsl
from threading import Thread

_pending_inputs = collections.deque()
_active_inputs = []
_texture_units = iter([])


def _init_slots():
    global _texture_units
    max_texture_image_units = pointer(c_uint())
    glsl.glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, max_texture_image_units)
    value = max_texture_image_units.contents.value
    _texture_units = iter([i for i in range(value if value > 0 else 16)])


def _input_devices():
    for input in _active_inputs:
        if isinstance(input, MultiInput):
            for i in input.inputs:
                if isinstance(i, InputDevice):
                    yield i
        elif isinstance(input, InputDevice):
            yield input


def _evdev_event(input):
    try:
        while True:
            try:
                for ev in input.dev.events():
                    input.handler.event(ev, target=input)
            except EventsDroppedException:
                for ev in input.dev.sync():
                    input.handler.event(ev, target=input)
    except IOError as e:
        if e.errno == ENODEV:
            print(f'input device {input.dev.name} unplugged')
        else:
            print(f'error reading events from input device {input.dev.name}', e)
    except Exception as e:
        print(f'error reading events from input device {input.dev.name}', e)
    finally:
        ClosingDevice(input)


def _validate_input(input, program, width, height):
    # Remove the input if its device has closed
    if isinstance(input, ClosingDevice):
        if (isinstance(input.target, Mouse)
                and (multi := next(filter(lambda i: isinstance(i, MultiMouse), _active_inputs), None))):
            multi.remove(input.target)
            if len(multi.mice) == 1:
                mouse = multi.mice[0]
                multi.remove(mouse)
                _active_inputs.append(mouse)
                _active_inputs.remove(multi)
        else:
            _active_inputs.remove(input.target)
        return

    # Check if it's an input device that's already open
    if (isinstance(input, InputDevice)
            and next(filter(lambda i: i.dev.fd.name == input.dev.fd.name, _input_devices()), None)):
        return

    # Hacky way to append the synthetic input at the end of the queue, so it's popped right next
    def _push_right(f): i = f(); _pending_inputs.rotate(-1); return i

    # Check the input refers to an existing uniform
    try:
        input.init(program=program, width=width, height=height)
    except NoActiveUniformVariable as e:
        if isinstance(input, Mouse) and input.name != 'iMouse':
            print(f"invalid {type(input).__name__} input '{input.name}': {e}")
        elif isinstance(input, Touchscreen):
            # TODO: keep the touchscreen device that's the same as the display device
            # Fall back to using the touchscreen as a mouse device
            _push_right(lambda: TouchMouse('iMouse', input.dev))
        elif isinstance(input, Trackpad):
            # Fall back to using the trackpad as a mouse device
            _push_right(lambda: TrackMouse('iMouse', input.dev))
        return
    except Exception as e:
        print(f"invalid {type(input).__name__} input '{input.name}': {e}")
        return

    # Handle the input multiplexing
    if isinstance(input, Mouse):
        # Multiplex mouse devices into a single input
        # TODO: multiplex by uniform name
        if multi := next(filter(lambda i: isinstance(i, MultiMouse), _active_inputs), None):
            multi.add(input)
        elif mouse := next(filter(lambda i: isinstance(i, Mouse), _active_inputs), None):
            _active_inputs.remove(mouse)
            _push_right(lambda: MultiMouse('iMouse')).add(mouse, input)
        else:
            _active_inputs.append(input)
    else:
        _active_inputs.append(input)

    # Start processing events from the input device
    if isinstance(input, InputDevice):
        input.dev.grab()
        Thread(target=_evdev_event, args=[input], daemon=True).start()


def _drain(q: collections.deque):
    while True:
        try:
            yield q.pop()
        except IndexError:
            break


@CFUNCTYPE(None, c_uint, c_uint, c_uint)
def _setup(program, width, height):
    _init_slots()

    # Drain all the inputs defined during initialisation
    for input in _drain(_pending_inputs):
        _validate_input(input, program, width, height)


@CFUNCTYPE(None, c_uint64, c_float)
def _update(frame, time):
    # Drain pending inputs (added at runtime)
    if len(_pending_inputs):
        program = c_uint()
        glsl.glGetIntegerv(GL_CURRENT_PROGRAM, pointer(program))

        viewport = (c_uint*4)()
        glsl.glGetIntegerv(GL_VIEWPORT, viewport)
        (width, height) = viewport[2:4]

        for input in _drain(_pending_inputs):
            _validate_input(input, program, width, height)

    # Render active inputs
    for input in _active_inputs:
        input.render(frame=frame, time=time)


glsl.onInit(_setup)
glsl.onRender(_update)


class NoActiveUniformVariable(Exception):
    name = None

    def __init__(self, name):
        super().__init__(f"no active uniform variable '{name}'")
        self.name = name


class Input:
    name = ''
    loc = None

    def __init__(self, name):
        self.name = name
        _pending_inputs.appendleft(self)

    def init(self, program, width, height):
        self.loc = glsl.glGetUniformLocation(program, bytes(self.name, 'utf-8'))
        if self.loc < 0:
            raise NoActiveUniformVariable(self.name)

    def render(self, frame, time):
        return


class Texture(Input):
    tex = None
    unit = None

    def init(self, **kwargs):
        super().init(**kwargs)

        self.tex = c_uint()
        self.unit = next(_texture_units)
        glsl.glUniform1i(self.loc, self.unit)
        glsl.glActiveTexture(GL_TEXTURE0 + self.unit)
        glsl.glGenTextures(1, pointer(self.tex))


class ImageTexture(Texture):
    path = ''
    transpose = None

    def __init__(self, name, path, transpose=None):
        super().__init__(name)
        self.path = path
        self.transpose = transpose

    def init(self, **kwargs):
        super().init(**kwargs)

        glsl.glBindTexture(GL_TEXTURE_2D, self.tex)
        glsl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glsl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glsl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glsl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        image = Image.open(self.path)
        if self.transpose:
            image = image.transpose(getattr(Image, self.transpose))
        data = image.convert('RGBA').tobytes()
        glsl.glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        glsl.glGenerateMipmap(GL_TEXTURE_2D)
        image.close()


class VolumeTexture(Texture):
    path = ''

    def __init__(self, name, path):
        super().__init__(name)
        self.path = path

    def init(self, **kwargs):
        super().init(**kwargs)

        data = Path(self.path).read_bytes()
        width = int.from_bytes(data[4:8], byteorder='little')
        height = int.from_bytes(data[8:12], byteorder='little')
        depth = int.from_bytes(data[12:16], byteorder='little')
        channels = int.from_bytes(data[16:17], byteorder='little')
        bin_format = int.from_bytes(data[18:20], byteorder='little')
        is_float = bin_format == 10

        internal_format = GL_RGBA
        source_format = GL_RGBA

        if channels == 4:
            internal_format = GL_RGBA16F if is_float else GL_RGBA8
            source_format = GL_RGBA
        elif channels == 3:
            internal_format = GL_RGB16F if is_float else GL_RGB8
            source_format = GL_RGB
        elif channels == 2:
            internal_format = GL_RG16F if is_float else GL_RG8
            source_format = GL_RG
        elif channels == 1:
            internal_format = GL_R16F if is_float else GL_R8
            source_format = GL_RED

        glsl.glBindTexture(GL_TEXTURE_3D, self.tex)
        glsl.glTexImage3D(GL_TEXTURE_3D, 0, internal_format, width, height, depth, 0, source_format,
                          GL_FLOAT if is_float else GL_UNSIGNED_BYTE, data[20:])
        glsl.glGenerateMipmap(GL_TEXTURE_3D)


class CubemapTexture(Texture):
    path = ''

    def __init__(self, name, path):
        super().__init__(name)
        self.path = path

    def init(self, **kwargs):
        super().init(**kwargs)

        glsl.glBindTexture(GL_TEXTURE_CUBE_MAP, self.tex)
        glsl.glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glsl.glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glsl.glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
        glsl.glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glsl.glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        image = Image.open(self.path)
        data = image.convert('RGB').tobytes()
        channels = len(image.getbands())
        for i in range(0, 6):
            glsl.glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB8, image.width, image.width, 0, GL_RGB,
                              GL_UNSIGNED_BYTE, data[i*image.width**2*channels:])
        glsl.glGenerateMipmap(GL_TEXTURE_CUBE_MAP)
        image.close()


class EventHandler:

    def event(self, ev, target, **__):
        raise NotImplementedError


class InputDevice(Input, EventHandler):
    dev = None
    handler = None

    def __init__(self, name, dev):
        super().__init__(name)
        self.dev = dev
        self.handler = self

    @property
    def device(self):
        return self.dev


class ClosingDevice(InputDevice):
    target: InputDevice = None

    def __init__(self, target):
        super().__init__(target.name, target.dev)
        self.target = target

    @property
    def device(self):
        return self.target.dev


keycodes = {
    EV_KEY.KEY_BACKSPACE: 8,
    EV_KEY.KEY_TAB: 9,
    EV_KEY.KEY_ENTER: 13,
    EV_KEY.KEY_LEFTSHIFT: 16,
    EV_KEY.KEY_RIGHTSHIFT: 16,
    EV_KEY.KEY_LEFTCTRL: 17,
    EV_KEY.KEY_RIGHTCTRL: 17,
    EV_KEY.KEY_LEFTALT: 18,
    EV_KEY.KEY_RIGHTALT: 18,
    EV_KEY.KEY_ESC: 27,
    EV_KEY.KEY_SPACE: 32,
    EV_KEY.KEY_LEFT: 37,
    EV_KEY.KEY_UP: 38,
    EV_KEY.KEY_RIGHT: 39,
    EV_KEY.KEY_DOWN: 40,
    EV_KEY.KEY_0: 48,
    EV_KEY.KEY_1: 49,
    EV_KEY.KEY_2: 50,
    EV_KEY.KEY_3: 51,
    EV_KEY.KEY_4: 52,
    EV_KEY.KEY_5: 53,
    EV_KEY.KEY_6: 54,
    EV_KEY.KEY_7: 55,
    EV_KEY.KEY_8: 56,
    EV_KEY.KEY_9: 57,
    EV_KEY.KEY_A: 65,
    EV_KEY.KEY_B: 66,
    EV_KEY.KEY_C: 67,
    EV_KEY.KEY_D: 68,
    EV_KEY.KEY_E: 69,
    EV_KEY.KEY_F: 70,
    EV_KEY.KEY_G: 71,
    EV_KEY.KEY_H: 72,
    EV_KEY.KEY_I: 73,
    EV_KEY.KEY_J: 74,
    EV_KEY.KEY_K: 75,
    EV_KEY.KEY_L: 76,
    EV_KEY.KEY_M: 77,
    EV_KEY.KEY_N: 78,
    EV_KEY.KEY_O: 79,
    EV_KEY.KEY_P: 80,
    EV_KEY.KEY_Q: 81,
    EV_KEY.KEY_R: 82,
    EV_KEY.KEY_S: 83,
    EV_KEY.KEY_T: 84,
    EV_KEY.KEY_U: 85,
    EV_KEY.KEY_V: 86,
    EV_KEY.KEY_W: 87,
    EV_KEY.KEY_X: 88,
    EV_KEY.KEY_Y: 89,
    EV_KEY.KEY_Z: 90,
    EV_KEY.KEY_LEFTMETA: 91,
    EV_KEY.KEY_RIGHTMETA: 92,
    EV_KEY.KEY_SLASH: 191,
}


class Keyboard(InputDevice, Texture):
    buffer = [0] * 256 * 3

    def event(self, ev, **_):
        if not ev.matches(EV_KEY):
            return
        code = -1
        if ev.code in keycodes:
            code = keycodes[ev.code]
        if code < 0:
            return

        if ev.value == 0:
            self.buffer[code] = 0
            self.buffer[256 + code] = 0
        elif ev.value == 1:
            self.buffer[code] = 255
            self.buffer[256 + code] = 255
            self.buffer[256 * 2 + code] = 255 - self.buffer[256 * 2 + code]

            # The keyboard device has been grabbed, so events are not sent to virtual
            # devices. The main thread has to be interrupted on CTRL+C explicitly.
            if ev.matches(EV_KEY.KEY_C) and self.buffer[17]:
                signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)

    def render(self, frame, **_):
        glsl.glActiveTexture(GL_TEXTURE0 + self.unit)
        glsl.glBindTexture(GL_TEXTURE_2D, self.tex)
        glsl.glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, 256, 3, 0, GL_RED, GL_UNSIGNED_BYTE,
                          (c_ubyte * (256 * 3))(*self.buffer))
        glsl.glGenerateMipmap(GL_TEXTURE_2D)
        for i in range(256, 2 * 256):
            self.buffer[i] = 0


class Mouse(InputDevice):
    drag = False
    resolution: (int, int) = None

    def init(self, width, height, **kwargs):
        super().init(width=width, height=height, **kwargs)

        self.resolution = (width, height)


class ButtonMouse(Mouse):
    click = False
    pointer_xy = drag_start = drag_xy = (1, 1)

    def event(self, ev, **_):
        if ev.matches(EV_KEY):
            if not ev.code == EV_KEY.BTN_LEFT:
                return
            if ev.value == 1:
                self.click = True
                self.drag_start = self.drag_xy = self.pointer_xy
                self.drag = True
            elif ev.value == 0:
                self.drag = False
        elif ev.matches(EV_REL):
            if ev.code == EV_REL.REL_X:
                self.pointer_xy = (max(1, min(self.pointer_xy[0] + ev.value, self.resolution[0])), self.pointer_xy[1])
            elif ev.code == EV_REL.REL_Y:
                self.pointer_xy = (self.pointer_xy[0], max(1, min(self.pointer_xy[1] - ev.value, self.resolution[1])))
            if self.drag:
                self.drag_xy = self.pointer_xy

    def render(self, **_):
        (z, w) = self.drag_start
        if self.drag:
            if not self.click:
                (z, w) = (z, -w)
        else:
            (z, w) = (-z, -w)

        glsl.glUniform4f(self.loc, c_float(self.drag_xy[0]), c_float(self.drag_xy[1]), c_float(z), c_float(w))

        if self.click:
            self.click = False


# TODO: rely on implicit Generic class once Python 3.12+ becomes a requirement
T = TypeVar('T')


class MultiInput(Generic[T], Input):
    inputs: [T] = []


class MultiMouse(MultiInput[Mouse], EventHandler):
    active: Optional[Mouse] = None

    @property
    def mice(self):
        return self.inputs

    def init(self, **kwargs):
        super().init(**kwargs)

        for mouse in self.mice:
            mouse.init(**kwargs)

    def add(self, *mice: [Mouse]):
        for mouse in mice:
            mouse.handler = self
            self.mice.append(mouse)

    def remove(self, *mice: [Mouse]):
        for mouse in mice:
            self.mice.remove(mouse)
            mouse.handler = mouse

    def event(self, ev, target, **kwargs):
        target.event(ev, **kwargs)
        if self.active is None and target.drag:
            self.active = target

    def render(self, **kwargs):
        if self.active is not None:
            self.active.render(**kwargs)
            if not self.active.drag:
                self.active = None


class _MTSlot:
    touch = False
    drag = (False, False)
    drag_xy = drag_start = (1, 1)


class Touchscreen(InputDevice):
    dev_abs_max = None
    resolution: (int, int) = None
    slots = []
    u4fv = None
    dirty = False

    def init(self, width, height, **kwargs):
        super().init(width=width, height=height, **kwargs)

        self.dev_abs_max = (self.dev.absinfo[EV_ABS.ABS_X].maximum, self.dev.absinfo[EV_ABS.ABS_Y].maximum)
        self.resolution = (width, height)
        self.slots = [_MTSlot() for _ in range(self.dev.num_slots)]
        self.u4fv = [0.0] * self.dev.num_slots * 4

    def event(self, ev, **_):
        self.dirty = True
        slot = self.slots[self.dev.current_slot]
        if ev.code == EV_ABS.ABS_MT_TRACKING_ID:
            if ev.value >= 0:
                slot.touch = True
            else:
                slot.drag = (False, False)
        elif ev.code == EV_ABS.ABS_MT_POSITION_X:
            slot.drag_xy = (ev.value / self.dev_abs_max[0] * self.resolution[0], slot.drag_xy[1])
            if not slot.drag[0]:
                slot.drag_start = (slot.drag_xy[0], slot.drag_start[1])
                slot.drag = (True, slot.drag[1])
        elif ev.code == EV_ABS.ABS_MT_POSITION_Y:
            slot.drag_xy = (slot.drag_xy[0], self.resolution[1] - ev.value / self.dev_abs_max[1] * self.resolution[1])
            if not slot.drag[1]:
                slot.drag_start = (slot.drag_start[0], slot.drag_xy[1])
                slot.drag = (slot.drag[0], True)

    def render(self, **_):
        if not self.dirty:
            return

        dirty = self.dirty
        self.dirty = False

        for i, slot in enumerate(self.slots):
            self.u4fv[4 * i] = slot.drag_xy[0]
            self.u4fv[4 * i + 1] = slot.drag_xy[1]
            (z, w) = slot.drag_start
            if slot.drag[0] and slot.drag[1]:
                if not slot.touch:
                    (z, w) = (z, -w)
            else:
                (z, w) = (-z, -w)
            self.u4fv[4 * i + 2] = z
            self.u4fv[4 * i + 3] = w
            if slot.touch:
                slot.touch = False
                self.dirty = True

        if dirty:
            glsl.glUniform4fv(self.loc, len(self.u4fv), (c_float * len(self.u4fv))(*self.u4fv))


class TouchMouse(Mouse):
    dev_abs_max = None
    _drag = (False, False)
    drag_xy = drag_start = (1, 1)
    touch = False

    def init(self, **kwargs):
        super().init(**kwargs)

        self.dev_abs_max = (self.dev.absinfo[EV_ABS.ABS_X].maximum, self.dev.absinfo[EV_ABS.ABS_Y].maximum)

    def event(self, ev, **_):
        if ev.code == EV_KEY.BTN_TOUCH:
            if ev.value == 1:
                self.drag = True
                self.touch = True
            else:
                self.drag = False
                self._drag = (False, False)
        elif ev.code == EV_ABS.ABS_X:
            self.drag_xy = (ev.value / self.dev_abs_max[0] * self.resolution[0], self.drag_xy[1])
            if not self._drag[0]:
                self.drag_start = (self.drag_xy[0], self.drag_start[1])
                self._drag = (True, self._drag[1])
        elif ev.code == EV_ABS.ABS_Y:
            self.drag_xy = (self.drag_xy[0], self.resolution[1] - ev.value / self.dev_abs_max[1] * self.resolution[1])
            if not self._drag[1]:
                self.drag_start = (self.drag_start[0], self.drag_xy[1])
                self._drag = (self._drag[0], True)

    def render(self, **_):
        (z, w) = self.drag_start
        if self.drag:
            if not self.touch:
                (z, w) = (z, -w)
        else:
            (z, w) = (-z, -w)

        glsl.glUniform4f(self.loc, c_float(self.drag_xy[0]), c_float(self.drag_xy[1]), c_float(z), c_float(w))

        if self.touch:
            self.touch = False


class Trackpad(InputDevice):
    dev_abs_x: (int, int) = None
    dev_abs_y: (int, int) = None
    resolution: (int, int) = None
    slots = []
    u4fv = None
    dirty = False

    def init(self, width, height, **kwargs):
        super().init(width=width, height=height, **kwargs)

        self.dev_abs_x = (self.dev.absinfo[EV_ABS.ABS_X].minimum, self.dev.absinfo[EV_ABS.ABS_X].maximum)
        self.dev_abs_y = (self.dev.absinfo[EV_ABS.ABS_Y].minimum, self.dev.absinfo[EV_ABS.ABS_Y].maximum)
        self.resolution = (width, height)
        self.slots = [_MTSlot() for _ in range(self.dev.num_slots)]
        self.u4fv = [0.0] * self.dev.num_slots * 4

    def event(self, ev, **_):
        self.dirty = True
        slot = self.slots[self.dev.current_slot]
        if ev.code == EV_ABS.ABS_MT_TRACKING_ID:
            if ev.value >= 0:
                slot.touch = True
            else:
                slot.drag = (False, False)
        elif ev.code == EV_ABS.ABS_MT_POSITION_X:
            slot.drag_xy = ((ev.value - self.dev_abs_x[0]) /
                            (self.dev_abs_x[1] - self.dev_abs_x[0]) * self.resolution[0], slot.drag_xy[1])
            if not slot.drag[0]:
                slot.drag_start = (slot.drag_xy[0], slot.drag_start[1])
                slot.drag = (True, slot.drag[1])
        elif ev.code == EV_ABS.ABS_MT_POSITION_Y:
            slot.drag_xy = (slot.drag_xy[0], self.resolution[1] - (ev.value - self.dev_abs_y[0]) /
                            (self.dev_abs_y[1] - self.dev_abs_y[0]) * self.resolution[1])
            if not slot.drag[1]:
                slot.drag_start = (slot.drag_start[0], slot.drag_xy[1])
                slot.drag = (slot.drag[0], True)

    def render(self, **_):
        if not self.dirty:
            return

        dirty = self.dirty
        self.dirty = False

        for i, slot in enumerate(self.slots):
            self.u4fv[4 * i] = slot.drag_xy[0]
            self.u4fv[4 * i + 1] = slot.drag_xy[1]
            (z, w) = slot.drag_start
            if slot.drag[0] and slot.drag[1]:
                if not slot.touch:
                    (z, w) = (z, -w)
            else:
                (z, w) = (-z, -w)
            self.u4fv[4 * i + 2] = z
            self.u4fv[4 * i + 3] = w
            if slot.touch:
                slot.touch = False
                self.dirty = True

        if dirty:
            glsl.glUniform4fv(self.loc, len(self.u4fv), (c_float * len(self.u4fv))(*self.u4fv))


class TrackMouse(Mouse):
    dev_abs_x: (int, int) = None
    dev_abs_y: (int, int) = None
    resolution: (int, int) = None
    _drag = (False, False)
    drag_xy = drag_start = (1, 1)
    touch = False

    def init(self, width, height, **kwargs):
        super().init(width=width, height=height, **kwargs)

        self.dev_abs_x = (self.dev.absinfo[EV_ABS.ABS_X].minimum, self.dev.absinfo[EV_ABS.ABS_X].maximum)
        self.dev_abs_y = (self.dev.absinfo[EV_ABS.ABS_Y].minimum, self.dev.absinfo[EV_ABS.ABS_Y].maximum)
        self.resolution = (width, height)

    def event(self, ev, **_):
        if ev.code == EV_KEY.BTN_TOUCH:
            if ev.value == 1:
                self.drag = True
                self.touch = True
            else:
                self.drag = False
                self._drag = (False, False)
        elif ev.code == EV_ABS.ABS_X:
            self.drag_xy = ((ev.value - self.dev_abs_x[0]) /
                            (self.dev_abs_x[1] - self.dev_abs_x[0]) * self.resolution[0], self.drag_xy[1])
            if not self._drag[0]:
                self.drag_start = (self.drag_xy[0], self.drag_start[1])
                self._drag = (True, self._drag[1])
        elif ev.code == EV_ABS.ABS_Y:
            self.drag_xy = (self.drag_xy[0], self.resolution[1] - (ev.value - self.dev_abs_y[0]) /
                            (self.dev_abs_y[1] - self.dev_abs_y[0]) * self.resolution[1])
            if not self._drag[1]:
                self.drag_start = (self.drag_start[0], self.drag_xy[1])
                self._drag = (self._drag[0], True)

    def render(self, **_):
        (z, w) = self.drag_start
        if self.drag:
            if not self.touch:
                (z, w) = (z, -w)
        else:
            (z, w) = (-z, -w)

        glsl.glUniform4f(self.loc, c_float(self.drag_xy[0]), c_float(self.drag_xy[1]), c_float(z), c_float(w))

        if self.touch:
            self.touch = False
