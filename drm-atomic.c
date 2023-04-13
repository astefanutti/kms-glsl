/*
 * Copyright (c) 2017 Rob Clark <rclark@redhat.com>
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

static int add_plane_property(drmModeAtomicReq *req, uint32_t obj_id,
				const char *name, uint64_t value)
{
	struct plane *obj = drm.plane;
	unsigned int i;
	int prop_id = -1;

	for (i = 0 ; i < obj->props->count_props ; i++) {
		if (strcmp(obj->props_info[i]->name, name) == 0) {
			prop_id = obj->props_info[i]->prop_id;
			break;
		}
	}

	if (prop_id < 0) {
		printf("no plane property: %s\n", name);
		return -EINVAL;
	}

	return drmModeAtomicAddProperty(req, obj_id, prop_id, value);
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
//	printf("page flip event ocurred: %12.6f\n", sec + (usec / 1000000.0));
}

static int atomic_run(const struct gbm *gbm, const struct egl *egl)
{
	struct gbm_bo *bo = NULL;
	struct drm_fb *fb;
	uint32_t i = 0;
	uint32_t flags = DRM_MODE_PAGE_FLIP_EVENT | DRM_MODE_ATOMIC_NONBLOCK;
	uint64_t start_time, report_time, cur_time;
	int ret;

	drmEventContext evctx = {
			.version = 4,
			.page_flip_handler = on_pageflip_event
	};

	/* Allow a modeset change for the first commit only. */
	flags |= DRM_MODE_ATOMIC_ALLOW_MODESET;

	start_time = report_time = get_time_ns();

	while (i < drm.count) {
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
			double secs = elapsed_time / (double)NSEC_PER_SEC;
			unsigned frames = i - 1;  /* first frame ignored */
			printf("Rendered %u frames in %f sec (%f fps)\n",
				frames, secs, (double)frames/secs);
			report_time = cur_time;
		}

		/* Check for user input: */
		struct pollfd fdset[] = { {
			.fd = STDIN_FILENO,
			.events = POLLIN,
		} };
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

		ret = drmHandleEvent(drm.fd, &evctx);
		if (ret) {
			printf("failed to wait for page flip completion\n");
			return -1;
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
	double secs = elapsed_time / (double)NSEC_PER_SEC;
	unsigned frames = i - 1;  /* first frame ignored */
	printf("Rendered %u frames in %f sec (%f fps)\n",
		frames, secs, (double)frames/secs);

	dump_perfcntrs(frames, elapsed_time);

	return ret;
}

/* Pick a plane.. something that at a minimum can be connected to
 * the chosen crtc, but prefer primary plane.
 *
 * Seems like there is some room for a drmModeObjectGetNamedProperty()
 * type helper in libdrm..
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

const struct drm * init_drm_atomic(int fd, const char *mode_str,
		unsigned int vrefresh, unsigned int count)
{
	uint32_t plane_id;
	int ret;

	ret = init_drm(&drm, fd, mode_str, vrefresh, count);
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

	drm.run = atomic_run;

	return &drm;
}
