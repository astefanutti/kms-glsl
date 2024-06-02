import argparse
import libevdev

parser = argparse.ArgumentParser(description='Test input device')
parser.add_argument('path', metavar='PATH', type=str, help='the path to the input device')
args = parser.parse_args()


def print_event(ev):
    print("Event: time {}.{:06d}, ".format(ev.sec, ev.usec), end='')
    if ev.matches(libevdev.EV_SYN):
        if ev.matches(libevdev.EV_SYN.SYN_MT_REPORT):
            print("++++++++++++++ {} ++++++++++++".format(ev.code.name))
        elif ev.matches(libevdev.EV_SYN.SYN_DROPPED):
            print(">>>>>>>>>>>>>> {} >>>>>>>>>>>>".format(ev.code.name))
        else:
            print("-------------- {} ------------".format(ev.code.name))
    else:
        print("type {:02x} {} code {:03x} {:20s} value {:4d}".
              format(ev.type.value, ev.type.name, ev.code.value, ev.code.name, ev.value))


def print_capabilities(device):
    v = device.driver_version
    print("Input driver version is {}.{}.{}".format(v >> 16, (v >> 8) & 0xff, v & 0xff))
    print("Input device ID: bus {:#x} vendor {:#x} product {:#x} version {:#x}".
          format(device.id["bustype"], device.id["vendor"], device.id["product"], device.id["version"]))
    print("Input device name: {}".format(device.name))
    print("Supported events:")

    for t, cs in device.evbits.items():
        print("  Event type {} ({})".format(t.value, t.name))

        for c in cs:
            if t in [libevdev.EV_LED, libevdev.EV_SND, libevdev.EV_SW]:
                v = device.value[c]
                print("    Event code {} ({}) state {}".format(c.value, c.name, v))
            else:
                print("    Event code {} ({})".format(c.value, c.name))

            if t == libevdev.EV_ABS:
                a = device.absinfo[c]
                print("       {:10s} {:6d}".format('Value', a.value))
                print("       {:10s} {:6d}".format('Minimum', a.minimum))
                print("       {:10s} {:6d}".format('Maximum', a.maximum))
                print("       {:10s} {:6d}".format('Fuzz', a.fuzz))
                print("       {:10s} {:6d}".format('Flat', a.flat))
                print("       {:10s} {:6d}".format('Resolution', a.resolution))

    print("Properties:")
    for p in device.properties:
        print("  Property type {} ({})".format(p.value, p.name))


try:
    with open(args.path, "rb") as fd:
        dev = libevdev.Device(fd)
        print_capabilities(dev)
        print("################################\n"
              "#      Waiting for events      #\n"
              "################################")

        while True:
            try:
                for e in dev.events():
                    print_event(e)
            except libevdev.EventsDroppedException:
                for e in dev.sync():
                    print_event(e)

except KeyboardInterrupt:
    pass
except IOError as e:
    import errno
    if e.errno == errno.EACCES:
        print("Insufficient permissions to access {}".format(args.path))
    elif e.errno == errno.ENOENT:
        print("Device {} does not exist".format(args.path))
    else:
        raise e
