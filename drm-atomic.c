/*
 * Copyright (c) 2017 Rob Clark <rclark@redhat.com>
 * Copyright (c) 2019 NVIDIA Corporation
 * Copyright (c) 2020 Antonin Stefanutti <antonin.stefanutti@gmail.com>
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

#include <errno.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "common.h"
#include "drm-common.h"

static struct drm drm;

static int add_connector_property(drmModeAtomicReq *req, uint32_t obj_id,
                                  const char *name, uint64_t value)
{
	struct connector *obj = drm.connector;
	unsigned int i;
	int prop_id = 0;

	for (i = 0 ; i < obj->props->count_props ; i++) {
		if (strcmp(obj->props_info[i]->name, name) == 0) {
			prop_id = obj->props_info[i]->prop_id;
			break;
		}
	}

	if (prop_id < 0) {
		printf("no connector property: %s\n", name);
		return -EINVAL;
	}

	return drmModeAtomicAddProperty(req, obj_id, prop_id, value);
}

static int add_crtc_property(drmModeAtomicReq *req, uint32_t obj_id,
                             const char *name, uint64_t value)
{
	struct crtc *obj = drm.crtc;
	unsigned int i;
	int prop_id = -1;

	for (i = 0 ; i < obj->props->count_props ; i++) {
		if (strcmp(obj->props_info[i]->name, name) == 0) {
			prop_id = obj->props_info[i]->prop_id;
			break;
		}
	}

	if (prop_id < 0) {
		printf("no crtc property: %s\n", name);
		return -EINVAL;
	}

	return drmModeAtomicAddProperty(req, obj_id, prop_id, value);
}

static int find_plane_prop(const char *name, unsigned int *prop_idx)
{
	struct plane *obj = drm.plane;
	unsigned int i;

	for (i = 0; i < obj->props->count_props; i++) {
		if (strcmp(obj->props_info[i]->name, name) == 0) {
			break;
		}
	}

	if (i == obj->props->count_props) {
		printf("no plane property: %s\n", name);
		return -EINVAL;
	}

	*prop_idx = i;

	return 0;
}

static int add_plane_property(drmModeAtomicReq *req, uint32_t obj_id,
                              const char *name, uint64_t value)
{
	struct plane *obj = drm.plane;
	unsigned int prop_idx;
	int res = find_plane_prop(name, &prop_idx);
	const drmModePropertyRes *prop_info;

	if (res) return res;

	prop_info = obj->props_info[prop_idx];
	return drmModeAtomicAddProperty(req, obj_id, prop_info->prop_id, value);
}

static int get_plane_property_val(const char *name, uint64_t *val)
{
	struct plane *obj = drm.plane;
	unsigned int prop_idx;
	int res = find_plane_prop(name, &prop_idx);

	if (res) return res;

	*val = obj->props->prop_values[prop_idx];

	return 0;
}

static int get_plane_format_modifiers()
{
	struct plane *plane = drm.plane;
	drmModePropertyBlobPtr blob;
	struct drm_format_modifier_blob *format_blob;
	struct drm_format_modifier *modifiers;
	uint32_t *formats;
	uint64_t blob_id;
	int res;
	uint32_t i;

	if ((res = get_plane_property_val("IN_FORMATS", &blob_id))) return res;

	blob = drmModeGetPropertyBlob(drm.fd, blob_id);

	if (!blob) {
		return -ENOMEM;
	}

	format_blob = blob->data;
	plane->formats = calloc(format_blob->count_formats,
	                        sizeof(plane->formats[0]));
	plane->modifiers = calloc(format_blob->count_modifiers,
	                          sizeof(plane->modifiers[0]));

	if (!plane->formats || !plane->modifiers) {
		free(plane->formats);
		free(plane->modifiers);
		return -ENOMEM;
	}

	formats = (uint32_t *) ((char *) format_blob +
	                        format_blob->formats_offset);
	modifiers = (struct drm_format_modifier *)
			((char *) format_blob + format_blob->modifiers_offset);

	memcpy(plane->formats, formats,
	       sizeof(plane->formats[0]) * format_blob->count_formats);

	for (i = 0; i < format_blob->count_modifiers; i++) {
		/*
		 * XXX Not quite right.  Should also stash which formats each
		 * modifier supports next to it in a per-modifier array, or
		 * vice-versa.
		 */
		plane->modifiers[i] = modifiers[i].modifier;
	}

	plane->num_formats = format_blob->count_formats;
	plane->num_modifiers = format_blob->count_modifiers;

	return 0;
}

static int drm_atomic_commit(uint32_t fb_id, uint32_t flags)
{
	drmModeAtomicReq *req;
	uint32_t plane_id = drm.plane->plane->plane_id;
	uint32_t blob_id;
	int ret;

	req = drmModeAtomicAlloc();

	if (flags & DRM_MODE_ATOMIC_ALLOW_MODESET) {
		if (add_connector_property(req, drm.connector_id, "CRTC_ID",
						drm.crtc_id) < 0)
				return -1;

		if (drmModeCreatePropertyBlob(drm.fd, drm.mode, sizeof(*drm.mode),
						&blob_id) != 0)
			return -1;

		if (add_crtc_property(req, drm.crtc_id, "MODE_ID", blob_id) < 0)
			return -1;

		if (add_crtc_property(req, drm.crtc_id, "ACTIVE", 1) < 0)
			return -1;
	}

	add_plane_property(req, plane_id, "FB_ID", fb_id);
	add_plane_property(req, plane_id, "CRTC_ID", drm.crtc_id);
	add_plane_property(req, plane_id, "SRC_X", 0);
	add_plane_property(req, plane_id, "SRC_Y", 0);
	add_plane_property(req, plane_id, "SRC_W", drm.mode->hdisplay << 16);
	add_plane_property(req, plane_id, "SRC_H", drm.mode->vdisplay << 16);
	add_plane_property(req, plane_id, "CRTC_X", 0);
	add_plane_property(req, plane_id, "CRTC_Y", 0);
	add_plane_property(req, plane_id, "CRTC_W", drm.mode->hdisplay);
	add_plane_property(req, plane_id, "CRTC_H", drm.mode->vdisplay);

	ret = drmModeAtomicCommit(drm.fd, req, flags, NULL);

	drmModeAtomicFree(req);

	return ret;
}

static void on_pageflip_event(
		int fd,
		unsigned int frame,
		unsigned int sec,
		unsigned int usec,
		void *userdata
) {
//	printf("page flip event occurred: %12.6f\n", sec + (usec / 1000000.0));
}

static int atomic_run(const struct gbm *gbm, const struct egl *egl)
{
	struct gbm_bo *bo = NULL;
	struct drm_fb *fb;
	uint32_t i = 0;
	uint64_t start_time, report_time, cur_time;
	int ret;

	uint32_t flags = DRM_MODE_ATOMIC_NONBLOCK;
	if (drm.async_page_flip) {
		flags |= DRM_MODE_PAGE_FLIP_ASYNC;
	} else {
		flags |= DRM_MODE_PAGE_FLIP_EVENT;
	}

	drmEventContext evctx = {
			.version = 4,
			.page_flip_handler = on_pageflip_event
	};

	/* Allow a modeset change for the first commit only. */
	flags |= DRM_MODE_ATOMIC_ALLOW_MODESET;

	start_time = report_time = get_time_ns();

	while (drm.count == 0 || i < drm.count) {
		unsigned frame = i;
		struct gbm_bo *next_bo;

		/* Start fps measuring on second frame, to remove the time spent
		 * compiling shader, etc, from the fps:
		 */
		if (i == 1) {
			start_time = report_time = get_time_ns();
		}

		if (!gbm->surface) {
			glBindFramebuffer(GL_FRAMEBUFFER, egl->fbs[frame % NUM_BUFFERS].fb);
		}

		egl->draw(start_time, i++);

		/* Block until all the buffered GL operations are completed.
		 * This is required on NVIDIA GPUs, for which the DRM drivers
		 * do not wait for the rendering to complete, upon executing
		 * page flipping operations.
		 */
		glFinish();

		if (gbm->surface) {
			eglSwapBuffers(egl->display, egl->surface);
		}

		if (gbm->surface) {
			next_bo = gbm_surface_lock_front_buffer(gbm->surface);
		} else {
			next_bo = gbm->bos[frame % NUM_BUFFERS];
		}
		if (!next_bo) {
			printf("Failed to lock frontbuffer\n");
			return -1;
		}
		fb = drm_fb_get_from_bo(next_bo);
		if (!fb) {
			printf("Failed to get a new framebuffer BO\n");
			return -1;
		}

		cur_time = get_time_ns();
		if (cur_time > (report_time + 2 * NSEC_PER_SEC)) {
			double elapsed_time = cur_time - start_time;
			double secs = elapsed_time / (double) NSEC_PER_SEC;
			unsigned frames = i - 1;  /* first frame ignored */
			printf("Rendered %u frames in %f sec (%f fps)\n",
			       frames, secs, (double) frames / secs);
			report_time = cur_time;
		}

		/* Check for user input: */
		struct pollfd fdset[] = {
				{
						.fd = STDIN_FILENO,
						.events = POLLIN,
				}
		};
		ret = poll(fdset, ARRAY_SIZE(fdset), 0);
		if (ret > 0) {
			printf("user interrupted!\n");
			return 0;
		}

		/*
		 * Here you could also update drm plane layers if you want
		 * hw composition
		 */
		ret = drm_atomic_commit(fb->fb_id, flags);
		if (ret) {
			printf("failed to commit: %s\n", strerror(errno));
			return -1;
		}

		if (!drm.async_page_flip) {
			ret = drmHandleEvent(drm.fd, &evctx);
			if (ret) {
				printf("failed to wait for page flip completion\n");
				return -1;
			}
		}

		/* release last buffer to render on again: */
		if (bo && gbm->surface)
			gbm_surface_release_buffer(gbm->surface, bo);
		bo = next_bo;

		/* Allow a modeset change for the first commit only. */
		flags &= ~(DRM_MODE_ATOMIC_ALLOW_MODESET);
	}

	finish_perfcntrs();

	cur_time = get_time_ns();
	double elapsed_time = cur_time - start_time;
	double secs = elapsed_time / (double) NSEC_PER_SEC;
	unsigned frames = i - 1;  /* first frame ignored */
	printf("Rendered %u frames in %f sec (%f fps)\n",
	       frames, secs, (double) frames / secs);

	dump_perfcntrs(frames, elapsed_time);

	return ret;
}

/* Pick a plane, something that at a minimum can be connected to
 * the chosen crtc, but prefer primary plane.
 *
 * Seems like there is some room for a drmModeObjectGetNamedProperty()
 * type helper in libdrm.
 */
static int get_plane_id(void)
{
	drmModePlaneResPtr plane_resources;
	uint32_t i, j;
	int ret = -EINVAL;
	int found_primary = 0;

	plane_resources = drmModeGetPlaneResources(drm.fd);
	if (!plane_resources) {
		printf("drmModeGetPlaneResources failed: %s\n", strerror(errno));
		return -1;
	}

	for (i = 0; (i < plane_resources->count_planes) && !found_primary; i++) {
		uint32_t id = plane_resources->planes[i];
		drmModePlanePtr plane = drmModeGetPlane(drm.fd, id);
		if (!plane) {
			printf("drmModeGetPlane(%u) failed: %s\n", id, strerror(errno));
			continue;
		}

		if (plane->possible_crtcs & (1 << drm.crtc_index)) {
			drmModeObjectPropertiesPtr props =
				drmModeObjectGetProperties(drm.fd, id, DRM_MODE_OBJECT_PLANE);

			/* primary or not, this plane is good enough to use: */
			ret = id;

			for (j = 0; j < props->count_props; j++) {
				drmModePropertyPtr p =
					drmModeGetProperty(drm.fd, props->props[j]);

				if ((strcmp(p->name, "type") == 0) &&
						(props->prop_values[j] == DRM_PLANE_TYPE_PRIMARY)) {
					/* found our primary plane, lets use that: */
					found_primary = 1;
				}

				drmModeFreeProperty(p);
			}

			drmModeFreeObjectProperties(props);
		}

		drmModeFreePlane(plane);
	}

	drmModeFreePlaneResources(plane_resources);

	return ret;
}

const struct drm * init_drm_atomic(int fd, const struct options *options)
{
	int ret;
	uint32_t plane_id;

	ret = init_drm(&drm, fd, options);
	if (ret)
		return NULL;

	ret = drmSetClientCap(drm.fd, DRM_CLIENT_CAP_ATOMIC, 1);
	if (ret) {
		printf("no atomic modesetting support: %s\n", strerror(errno));
		return NULL;
	}

	ret = get_plane_id();
	if (!ret) {
		printf("could not find a suitable plane\n");
		return NULL;
	} else {
		plane_id = ret;
	}

	/* We only do single plane to single crtc to single connector, no
	 * fancy multi-monitor or multi-plane stuff.  So just grab the
	 * plane/crtc/connector property info for one of each:
	 */
	drm.plane = calloc(1, sizeof(*drm.plane));
	drm.crtc = calloc(1, sizeof(*drm.crtc));
	drm.connector = calloc(1, sizeof(*drm.connector));

#define get_resource(type, Type, id) do { 					\
		drm.type->type = drmModeGet##Type(drm.fd, id);			\
		if (!drm.type->type) {						\
			printf("could not get %s %i: %s\n",			\
					#type, id, strerror(errno));		\
			return NULL;						\
		}								\
	} while (0)

	get_resource(plane, Plane, plane_id);
	get_resource(crtc, Crtc, drm.crtc_id);
	get_resource(connector, Connector, drm.connector_id);

#define get_properties(type, TYPE, id) do {					\
		uint32_t i;							\
		drm.type->props = drmModeObjectGetProperties(drm.fd,		\
				id, DRM_MODE_OBJECT_##TYPE);			\
		if (!drm.type->props) {						\
			printf("could not get %s %u properties: %s\n", 		\
					#type, id, strerror(errno));		\
			return NULL;						\
		}								\
		drm.type->props_info = calloc(drm.type->props->count_props,	\
				sizeof(*drm.type->props_info));			\
		for (i = 0; i < drm.type->props->count_props; i++) {		\
			drm.type->props_info[i] = drmModeGetProperty(drm.fd,	\
					drm.type->props->props[i]);		\
		}								\
	} while (0)

	get_properties(plane, PLANE, plane_id);
	get_properties(crtc, CRTC, drm.crtc_id);
	get_properties(connector, CONNECTOR, drm.connector_id);

	get_plane_format_modifiers();

	drm.run = atomic_run;
	drm.async_page_flip = options->async_page_flip;

	return &drm;
}
