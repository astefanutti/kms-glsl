# KMS GLSL

KMS GLSL is a command line tool that runs OpenGL fragment shaders, using the [DRM/KMS Linux kernel subsystem](https://en.wikipedia.org/wiki/Direct_Rendering_Manager).
It runs shaders fullscreen, and does not require any windowing system, like X or Wayland.

It has initially been developed to run shaders from [Shadertoy](https://www.shadertoy.com), on the [Raspberry Pi](https://ttt.io/glsl-raspberry-pi), using the VC4/V3D KMS driver, that supports OpenGL ES 3.1 (as of January 2023).
However, it should work with any GPU and display controller hardware, provided a DRM/KMS driver is available.
It has been successfully tested with the [Jetson Nano](https://ttt.io/glsl-jetson-nano), as well as the Raspberry Pi Zero.

In the following picture, this [Shadertoy shader](https://www.shadertoy.com/view/MsX3Wj) runs on the Raspberry Pi 4, connected to the official Raspberry Pi 7" touchscreen monitor, in WVGA (800×480) resolution:

![A Shadertoy shader running on the Raspberry Pi 4](./raspberry_pi.jpg)

[Another shader](https://www.shadertoy.com/view/fstyD4) that runs on the Jetson Nano in full HD (1920×1080) resolution:

![A Shadertoy shader running on the Jetson Nano](./jetson_nano.jpg)

## Build

You need to clone the project, and run the following commands:

```shell
$ sudo apt update
# Install the build tools
$ sudo apt install gcc make
# Install the required DRM, GBM, EGL and OpenGL ES API headers
$ sudo apt install libdrm-dev libgbm-dev libegl-dev libgles2-mesa-dev
# Install the X C binding and RandR extension header / library files (optional)
$ sudo apt install libxcb-randr0-dev
# Build the glsl binary and library
$ make
```

## Usage

Once you've successfully built the binary / library, you can either run it directly, or use the Python wrapper, as explained below.

### Python

KMS GLSL comes with a Python wrapper around the native library, that adds a layer for managing shader inputs, that you can also extend to add your own inputs, as well as an augmented CLI:

```console
$ python glsl.py -h
usage: glsl.py [-h] [--async-page-flip | --no-async-page-flip]
               [--atomic-drm-mode | --no-atomic-drm-mode] [-D DEVICE] [--mode MODE]
               [-k UNIFORM] [--touch UNIFORM] [-c UNIFORM FILE] [-t UNIFORM FILE]
               [-v UNIFORM FILE] [-m <UNIFORM>.KEY VALUE]
               FILE

Run OpenGL shaders using DRM/KMS

positional arguments:
  FILE                  the shader file

optional arguments:
  -h, --help            show this help message and exit
  --async-page-flip, --no-async-page-flip
                        use async page flipping
  --atomic-drm-mode, --no-atomic-drm-mode
                        use atomic mode setting and fencing
  -D DEVICE, --device DEVICE
                        the DRM device
  --mode MODE           specify the video mode in the format
                        <resolution>[-<vrefresh>]
  -k UNIFORM, --keyboard UNIFORM
                        add keyboard
  --touch UNIFORM       add touch device
  -c UNIFORM FILE, --cubemap UNIFORM FILE
                        add cubemap
  -t UNIFORM FILE, --texture UNIFORM FILE
                        add texture
  -v UNIFORM FILE, --volume UNIFORM FILE
                        add volume
  -m <UNIFORM>.KEY VALUE, --metadata <UNIFORM>.KEY VALUE
                        set uniform metadata
```

You can try it with the shaders copied from Shadertoy available in the `examples` directory, e.g.:

```shell
$ python glsl.py examples/plasma_globe.glsl -t iChannel0 presets/tex_RGBA_noise_medium.png
```

Press <kbd>Ctrl</kbd>+<kbd>c</kbd> to exit the program.
You can explore [shadertoy.com](https://www.shadertoy.com) to find additional shaders.

> [!NOTE]
> [Shaders](https://www.shadertoy.com/howto#q1) from [Shadertoy](https://www.shadertoy.com/) are currently expected as input shader files.
>
> The shaders from the `examples` directory assume OpenGL ES 3.1 support, and may not work with lower versions of the specification.

You can find the documentation on how to add your own inputs in the `glsl.py` file.

### Native

If you cannot, or don't want to, use Python, you can directly use the native CLI:

```console
$ ./glsl -h
Usage: ./glsl [-aAcDfmpvx] <shader_file>

options:
    -a, --async              use async page flipping
    -A, --atomic             use atomic mode setting and fencing
    -c, --count              run for the specified number of frames
    -D, --device=DEVICE      use the given device
    -f, --format=FOURCC      framebuffer format
    -h, --help               print usage
    -m, --modifier=MODIFIER  hardcode the selected modifier
    -p, --perfcntr=LIST      sample specified performance counters using
                             the AMD_performance_monitor extension (comma
                             separated list)
    -v, --vmode=VMODE        specify the video mode in the format
                             <mode>[-<vrefresh>]
    -x, --surfaceless        use surfaceless mode, instead of GBM surface
```

Note no inputs can be provided using the native CLI directly.

## Raspberry Pi

Assuming you've installed [Raspberry Pi OS](https://www.raspberrypi.org/software/operating-systems), the DRM/KMS VC4/V3D driver can be configured by running the following commands:

1. Edit the [`/boot/config.txt`](https://www.raspberrypi.org/documentation/configuration/config-txt/) file, e.g.:

    ```shell
    $ sudo vi /boot/config.txt
    ```

2. Set the following properties:

    ```properties
    # Required: Enable the DRM/KMS VC4 V3D driver
    dtoverlay=vc4-kms-v3d
    # Optional: Increase the memory reserved for the GPU
    gpu_mem=128
    ```

3. Reboot your Raspberry Pi, so that the changes are taken into account, e.g.:

    ```shell
    $ sudo reboot
    ```

Note this also works with the Raspberry Pi OS Lite versions, as it does not need any compositor like X, Wayland or similar.

It has successfully been tested on the RPi 4, running the Raspberry Pi OS 2023-09-22, Linux kernel 5.15.

## Roadmap

- Add support for texture buffers
- Add support for audio / video inputs
- Parse GLSL files to retrieve uniforms metadata

## Credits

The DRM/KMS ceremony code is copied from [kmscube](https://gitlab.freedesktop.org/mesa/kmscube/).

The shader examples are copied from the [Shadertoy](https://www.shadertoy.com) website URLs commented at the top of each file.
