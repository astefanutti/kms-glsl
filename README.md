# KMS GLSL

KMS GLSL is a command line tool that runs OpenGL fragment shaders, using the [DRM/KMS Linux kernel subsystem](https://en.wikipedia.org/wiki/Direct_Rendering_Manager).
It runs shaders fullscreen, and does not require any windowing system, like X or Wayland.

It has initially been developed to run shaders from [Shadertoy](https://www.shadertoy.com), on the [Raspberry Pi](#raspberry-pi), using the VC4/V3D KMS driver, that exposes OpenGL ES 3.1 (as of December 2020).
However, it should work with any GPU and display controller hardware, provided a DRM/KMS driver is available.

In the following picture, this [Shadertoy shader](https://www.shadertoy.com/view/MsX3Wj) runs on the official Raspberry Pi 7″ touchscreen monitor, connected to a Raspberry Pi 4<sup name="a1">[1](#f1)</sup>:

![A Shadertoy shader running on a Raspberry Pi 4](./torus.jpg)

## Build

```shell
$ sudo apt update
# Install the build tools
$ sudo apt install gcc make
# Install the required DRM, GBM, EGL and OpenGL ES API headers
$ sudo apt install libdrm-dev libgbm-dev libegl-dev libgles2-mesa-dev
# Install the X C binding and RandR extension header / library files (optional)
$ sudo apt install libxcb-randr0-dev
# Build the glsl CLI binary
$ make
```

## Usage

```console
$ ./glsl -h
Usage: ./glsl [-AcDfmpvx] <shader_file>

options:
    -A, --atomic             use atomic modesetting and fencing
    -c, --count              run for the specified number of frames
    -D, --device=DEVICE      use the given device
    -f, --format=FOURCC      framebuffer format
    -h, --help      		 print usage
    -m, --modifier=MODIFIER  hardcode the selected modifier
    -p, --perfcntr=LIST      sample specified performance counters using
                             the AMD_performance_monitor extension (comma
                             separated list)
    -v, --vmode=VMODE        specify the video mode in the format
                             <mode>[-<vrefresh>]
    -x, --surfaceless        use surfaceless mode, instead of GBM surface
```

> :warning: [Image shaders](https://www.shadertoy.com/howto#q1) from [Shadertoy](https://www.shadertoy.com/) are currently expected as input shader files.

You can try with the shaders copied from Shadertoy into the `examples` directory, e.g.:

```shell
$ ./glsl examples/stripey_torus_interior.glsl
```

Press <kbd>Ctrl</kbd>+<kbd>c</kbd> to exit the program.
You can explore [shadertoy.com](https://www.shadertoy.com) to find additional shaders.

## Raspberry Pi

Assuming you've installed [Raspberry Pi OS](https://www.raspberrypi.org/software/operating-systems), the DRM/KMS VC4/V3D driver can be configured by running the following commands:

1. Edit the [`/boot/config.txt`](https://www.raspberrypi.org/documentation/configuration/config-txt/) file, e.g.:

    ```shell
    $ sudo vi /boot/config.txt
    ```

2. Set the following properties:

    ```properties
    # Required: Enable the firmware/fake DRM/KMS VC4/V3D driver
    dtoverlay=vc4-fkms-v3d
    # Optional: Increase the memory reserved for the GPU
    #           16MB disables certain GPU features
    gpu_mem=64
    # Optional: Avoid GPU down-clocking below 500 MHz that slows FPS down
    #           Should be set to 250 on the RPi 3
    v3d_freq_min=500
    ```

3. Reboot your Raspberry Pi, so that the changes are taken into account, e.g.:

    ```shell
    $ sudo reboot
    ```

Note this also works with the Raspberry Pi OS Lite versions, as it does not need any compositor like X, Wayland or similar.

It has successfully been tested on the RPi 3B+ and RPi 4, running the Raspberry Pi OS Lite 2020-12-02, Linux kernel 5.4.79.

## TODO

- Feed the mouse uniform with touchscreen events from [tslib](http://www.tslib.org)
- Forward keyboard events to the input keyboard texture
- Feed audio / video inputs to microphone / camera textures
- Package Shadertoy default media files
- Load arbitrary input texture files
- Ability to load multiple shaders into a carousel-like experience

## Credits

The DRM/KMS ceremony code is copied from [kmscube](https://gitlab.freedesktop.org/mesa/kmscube/).

The shader examples are copied from the [Shadertoy](https://www.shadertoy.com) website URLs commented at the top of each file.

---

<a name="f1">1</a>. The setup is described in the [Anatomy of my Kubernetes cluster](https://ttt.io/anatomy-of-my-kubernetes-cluster). [↩](#a1)
