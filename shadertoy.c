/*
 * Copyright © 2020 Antonin Stefanutti <antonin.stefanutti@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sub license,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#define _GNU_SOURCE

#include <assert.h>
#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <GLES3/gl3.h>

#include "common.h"

GLint iTime;

static const char *shadertoy_vs =
		"attribute vec3 position;                \n"
		"void main()                             \n"
		"{                                       \n"
		"    gl_Position = vec4(position, 1.0);  \n"
		"}                                       \n";

static const char *shadertoy_fs_tmpl =
		"precision mediump float;                                                             \n"
		"uniform vec3      iResolution;           // viewport resolution (in pixels)          \n"
		"uniform float     iTime;                 // shader playback time (in seconds)        \n"
		"uniform vec4      iMouse;                // mouse pixel coords                       \n"
		"uniform vec4      iDate;                 // (year, month, day, time in seconds)      \n"
		"uniform float     iSampleRate;           // sound sample rate (i.e., 44100)          \n"
		"uniform vec3      iChannelResolution[4]; // channel resolution (in pixels)           \n"
		"uniform float     iChannelTime[4];       // channel playback time (in sec)           \n"
		"                                                                                     \n"
		"%s                                                                                   \n"
		"                                                                                     \n"
		"void main()                                                                          \n"
		"{                                                                                    \n"
		"    mainImage(gl_FragColor, gl_FragCoord.xy);                                        \n"
		"}                                                                                    \n";

static const GLfloat vertices[] = {
		// First triangle:
		1.0f, 1.0f,
		-1.0f, 1.0f,
		-1.0f, -1.0f,
		// Second triangle:
		-1.0f, -1.0f,
		1.0f, -1.0f,
		1.0f, 1.0f,
};

static char *load_shader(const char *file) {
	struct stat statbuf;
	char *frag;
	int fd, ret;

	fd = open(file, 0);
	if (fd < 0) {
		err(fd, "could not open '%s'", file);
	}

	ret = fstat(fd, &statbuf);
	if (ret < 0) {
		err(ret, "could not stat '%s'", file);
	}

	const char *text = mmap(NULL, statbuf.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
	asprintf(&frag, shadertoy_fs_tmpl, text);

	return frag;
}

static void draw_shadertoy(unsigned i) {
	glUniform1f(iTime, (float) i / 60.0f);

	start_perfcntrs();

	glDrawArrays(GL_TRIANGLES, 0, 6);

	end_perfcntrs();
}

int init_shadertoy(const struct gbm *gbm, struct egl *egl, const char *file) {
	int ret;
	char *shadertoy_fs;
	GLuint program, vbo;
	GLint iResolution;

	shadertoy_fs = load_shader(file);

	ret = create_program(shadertoy_vs, shadertoy_fs);
	if (ret < 0) {
		printf("failed to create program\n");
		return -1;
	}

	program = ret;

	ret = link_program(program);
	if (ret) {
		printf("failed to link program\n");
		return -1;
	}

	glViewport(0, 0, gbm->width, gbm->height);
	glUseProgram(program);
	iTime = glGetUniformLocation(program, "iTime");
	iResolution = glGetUniformLocation(program, "iResolution");
	glUniform3f(iResolution, gbm->width, gbm->height, 0);
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), 0, GL_STATIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), &vertices[0]);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (const GLvoid *) (intptr_t) 0);
	glEnableVertexAttribArray(0);

	egl->draw = draw_shadertoy;

	return 0;
}
