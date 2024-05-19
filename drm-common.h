/*
 * Copyright (c) 2017 Rob Clark <rclark@redhat.com>
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

#ifndef _DRM_COMMON_H
#define _DRM_COMMON_H

#include <xf86drm.h>
#include <xf86drmMode.h>

struct plane {
	drmModePlane *plane;
	drmModeObjectProperties *props;
	drmModePropertyRes **props_info;
	uint32_t num_formats;
	uint32_t *formats;
	uint32_t num_modifiers;
	uint64_t *modifiers;
};

struct crtc {
	drmModeCrtc *crtc;
	drmModeObjectProperties *props;
	drmModePropertyRes **props_info;
};

struct connector {
	drmModeConnector *connector;
	drmModeObjectProperties *props;
	drmModePropertyRes **props_info;
};

struct drm {
	int fd;

	struct plane *plane;
	struct crtc *crtc;
	struct connector *connector;
	int crtc_index;

	drmModeModeInfo *mode;
	uint32_t crtc_id;
	uint32_t connector_id;

	bool async_page_flip;

	/* number of frames to run for: */
	unsigned int frames;

	int (*run)(const struct gbm *gbm, const struct egl *egl);
};

struct drm_fb {
	struct gbm_bo *bo;
	uint32_t fb_id;
};

struct drm_fb * drm_fb_get_from_bo(struct gbm_bo *bo);

int find_drm_device();

int find_plane_prop(const struct drm *drm, const char *name, unsigned int *prop_idx);

const uint64_t *get_drm_format_modifiers(const struct drm *drm, unsigned int *count);

int init_drm(struct drm *drm, int fd, const struct options *options);

const struct drm *init_drm_legacy(int fd, const struct options *options);

const struct drm *init_drm_atomic(int fd, const struct options *options);

#endif /* _DRM_COMMON_H */
