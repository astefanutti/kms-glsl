CC=gcc
CFLAGS=-c -g -Wall -O3 -Winvalid-pch -Wextra -std=gnu99 -fdiagnostics-color=always -pipe -pthread -I/usr/include/libdrm
LDFLAGS=-Wl,--as-needed -Wl,--no-undefined
LDLIBS=-lGLESv2 -lEGL -ldrm -lgbm -lxcb-randr -lxcb
SOURCES=common.c drm-atomic.c drm-common.c drm-legacy.c glsl.c lease.c perfcntrs.c shadertoy.c
OBJECTS=$(SOURCES:%.c=%.o)
EXECUTABLE=glsl

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) $(LDLIBS) -o $@

.c.o:
	$(CC) $(CFLAGS) $< -o $@

clean :
	rm -f *.o $(EXECUTABLE)
