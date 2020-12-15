/*
 * Copyright © 2020 Google, Inc.
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
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <GLES3/gl3.h>

#include "common.h"

/* Module to collect a specified set of performance counts, and accumulate
 * results, using the GL_AMD_performance_monitor extension.
 *
 * Call start_perfcntrs() before the draw(s) to measure, and end_perfcntrs()
 * after the last draw to measure.  This can be done multiple times, with
 * the results accumulated.
 */

/**
 * Accumulated counter result:
 */
union counter_result {
	uint32_t u32;   /* GL_UNSIGNED_INT */
	float    f;     /* GL_FLOAT, GL_PERCENTAGE_AMD */
	uint64_t u64;   /* GL_UNSIGNED_INT64_AMD */
};

/**
 * Tracking for a requested counter
 */
struct counter {
	union counter_result result;
	/* index into perfcntrs.groups[gidx].counters[cidx]
	 * Note that the group_idx/counter_idx is not necessarily the
	 * same as the group_id/counter_id.
	 */
	unsigned gidx;
	unsigned cidx;
};

/**
 * Description of gl counter groups and counters:
 */

struct gl_counter {
	char *name;
	GLuint counter_id;
	GLuint counter_type;
	struct counter *counter;  /* NULL if this is not a counter we track */
};

struct gl_counter_group {
	char *name;
	GLuint group_id;
	GLint max_active_counters;
	GLint num_counters;
	struct gl_counter *counters;

	/* number of counters in this group which are enabled: */
	int num_enabled_counters;
};

struct gl_monitor {
	GLuint id;
	bool valid;
	bool active;
};

/**
 * module state
 */
static struct {
	const struct egl *egl;

	/* The extension doesn't let us pause/resume a single counter, so
	 * instead use a sequence of monitors, one per start_perfcntrs()/
	 * end_perfcntrs() pair, so that we don't need to immediately read
	 * back a result, which could cause a stall.
	 */
	struct gl_monitor monitors[4];
	unsigned current_monitor;

	/* The requested counters to monitor:
	 */
	unsigned num_counters;
	struct counter *counters;

	/* The description of all counter groups and the counters they
	 * contain, not just including the ones we monitor.
	 */
	GLint num_groups;
	struct gl_counter_group *groups;

} perfcntr;

static void get_groups_and_counters(const struct egl *egl)
{
	int n;

	egl->glGetPerfMonitorGroupsAMD(&perfcntr.num_groups, 0, NULL);
	perfcntr.groups = calloc(perfcntr.num_groups, sizeof(struct gl_counter_group));

	GLuint group_ids[perfcntr.num_groups];
	egl->glGetPerfMonitorGroupsAMD(NULL, perfcntr.num_groups, group_ids);

	for (int i = 0; i < perfcntr.num_groups; i++) {
		struct gl_counter_group *g = &perfcntr.groups[i];

		g->group_id = group_ids[i];

		egl->glGetPerfMonitorGroupStringAMD(g->group_id, 0, &n, NULL);
		g->name = malloc(n+1);
		egl->glGetPerfMonitorGroupStringAMD(g->group_id, n+1, NULL, g->name);

		egl->glGetPerfMonitorCountersAMD(g->group_id, &g->num_counters,
			&g->max_active_counters, 0, NULL);

		g->counters = calloc(g->num_counters, sizeof(struct gl_counter));

		GLuint counter_ids[g->num_counters];
		egl->glGetPerfMonitorCountersAMD(g->group_id, NULL, NULL,
			g->num_counters, counter_ids);

		printf("GROUP[%u]: name=%s, max_active_counters=%u, num_counters=%u\n",
			g->group_id, g->name, g->max_active_counters, g->num_counters);

		for (int j = 0; j < g->num_counters; j++) {
			struct gl_counter *c = &g->counters[j];

			c->counter_id = counter_ids[j];

			egl->glGetPerfMonitorCounterStringAMD(g->group_id,
				c->counter_id, 0, &n, NULL);
			c->name = malloc(n+1);
			egl->glGetPerfMonitorCounterStringAMD(g->group_id,
				c->counter_id, n+1, NULL, c->name);

			egl->glGetPerfMonitorCounterInfoAMD(g->group_id,
				c->counter_id, GL_COUNTER_TYPE_AMD,
				&c->counter_type);

			printf("\tCOUNTER[%u]: name=%s, counter_type=%04x\n",
				c->counter_id, c->name, c->counter_type);
		}
	}
}

static void find_counter(const char *name, unsigned *group_idx, unsigned *counter_idx)
{
	for (int i = 0; i < perfcntr.num_groups; i++) {
		struct gl_counter_group *g = &perfcntr.groups[i];

		for (int j = 0; j < g->num_counters; j++) {
			struct gl_counter *c = &g->counters[j];

			if (strcmp(name, c->name) == 0) {
				*group_idx = i;
				*counter_idx = j;
				return;
			}
		}
	}

	errx(-1, "Could not find counter: %s", name);
}

static void add_counter(const char *name)
{
	int idx = perfcntr.num_counters++;

	perfcntr.counters = realloc(perfcntr.counters,
		perfcntr.num_counters * sizeof(struct counter));

	struct counter *c = &perfcntr.counters[idx];
	memset(c, 0, sizeof(*c));

	find_counter(name, &c->gidx, &c->cidx);

	struct gl_counter_group *g = &perfcntr.groups[c->gidx];
	if (g->num_enabled_counters >= g->max_active_counters) {
		errx(-1, "Too many counters in group '%s'", g->name);
	}

	g->num_enabled_counters++;
}

/* parse list of performance counter names, and find their group+counter */
static void find_counters(const char *perfcntrs)
{
	char *cnames, *s;

	cnames = strdup(perfcntrs);
	while ((s = strstr(cnames, ","))) {
		char *name = cnames;
		s[0] = '\0';
		cnames = &s[1];

		add_counter(name);
	}

	add_counter(cnames);
}

void init_perfcntrs(const struct egl *egl, const char *perfcntrs)
{
	if (egl_check(egl, glGetPerfMonitorGroupsAMD) ||
	    egl_check(egl, glGetPerfMonitorCountersAMD) ||
	    egl_check(egl, glGetPerfMonitorGroupStringAMD) ||
	    egl_check(egl, glGetPerfMonitorCounterStringAMD) ||
	    egl_check(egl, glGetPerfMonitorCounterInfoAMD) ||
	    egl_check(egl, glGenPerfMonitorsAMD) ||
	    egl_check(egl, glDeletePerfMonitorsAMD) ||
	    egl_check(egl, glSelectPerfMonitorCountersAMD) ||
	    egl_check(egl, glBeginPerfMonitorAMD) ||
	    egl_check(egl, glEndPerfMonitorAMD) ||
	    egl_check(egl, glGetPerfMonitorCounterDataAMD)) {
		errx(-1, "AMD_performance_monitor is not supported");
	}

	get_groups_and_counters(egl);
	find_counters(perfcntrs);

	/* setup enabled counters.. do this after realloc() stuff,
	 * otherwise the counter pointer may not be valid:
	 */
	for (unsigned i = 0; i < perfcntr.num_counters; i++) {
		struct counter *c = &perfcntr.counters[i];
		perfcntr.groups[c->gidx].counters[c->cidx].counter = c;
	}

	perfcntr.egl = egl;
}

/* Create perf-monitor, and configure the counters it will monitor */
static void init_monitor(struct gl_monitor *m)
{
	const struct egl *egl = perfcntr.egl;

	assert(!m->valid);
	assert(!m->active);

	egl->glGenPerfMonitorsAMD(1, &m->id);

	for (int i = 0; i < perfcntr.num_groups; i++) {
		struct gl_counter_group *g = &perfcntr.groups[i];

		if (!g->num_enabled_counters)
			continue;

		int idx = 0;
		GLuint counters[g->num_enabled_counters];

		for (int j = 0; j < g->num_counters; j++) {
			struct gl_counter *c = &g->counters[j];

			if (!c->counter)
				continue;

			assert(idx < g->num_enabled_counters);
			counters[idx++] = c->counter_id;
		}

		assert(idx == g->num_enabled_counters);
		egl->glSelectPerfMonitorCountersAMD(m->id, GL_TRUE,
			g->group_id, g->num_enabled_counters, counters);
	}

	m->valid = true;
}

static struct gl_counter *lookup_counter(GLuint group_id, GLuint counter_id)
{
	for (int i = 0; i < perfcntr.num_groups; i++) {
		struct gl_counter_group *g = &perfcntr.groups[i];

		if (g->group_id != group_id)
			continue;

		for (int j = 0; j < g->num_counters; j++) {
			struct gl_counter *c = &g->counters[j];

			if (c->counter_id != counter_id)
				continue;

			return c;
		}
	}

	errx(-1, "invalid counter: group_id=%u, counter_id=%u",
		group_id, counter_id);
}

/* Collect monitor results and delete monitor */
static void finish_monitor(struct gl_monitor *m)
{
	const struct egl *egl = perfcntr.egl;

	assert(m->valid);
	assert(!m->active);

	GLuint result_size;
	egl->glGetPerfMonitorCounterDataAMD(m->id, GL_PERFMON_RESULT_SIZE_AMD,
		sizeof(GLint), &result_size, NULL);

	GLuint *data = malloc(result_size);

	GLsizei bytes_written;
	egl->glGetPerfMonitorCounterDataAMD(m->id, GL_PERFMON_RESULT_AMD,
			result_size, data, &bytes_written);

	GLsizei idx = 0;
	while ((4 * idx) < bytes_written) {
		GLuint group_id = data[idx++];
		GLuint counter_id = data[idx++];

		struct gl_counter *c = lookup_counter(group_id, counter_id);

		assert(c->counter);

		switch(c->counter_type) {
		case GL_UNSIGNED_INT:
			c->counter->result.u32 += *(uint32_t *)(&data[idx]);
			idx += 1;
			break;
		case GL_FLOAT:
			c->counter->result.f += *(float *)(&data[idx]);
			idx += 1;
			break;
		case GL_UNSIGNED_INT64_AMD:
			c->counter->result.u64 += *(uint64_t *)(&data[idx]);
			idx += 2;
			break;
		case GL_PERCENTAGE_AMD:
		default:
			errx(-1, "TODO unhandled counter type: 0x%04x",
				c->counter_type);
			break;
		}
	}

	egl->glDeletePerfMonitorsAMD(1, &m->id);
	m->valid = false;
}

void start_perfcntrs(void)
{
	const struct egl *egl = perfcntr.egl;

	if (!egl) {
		return;
	}

	struct gl_monitor *m = &perfcntr.monitors[perfcntr.current_monitor];

	/* once we wrap-around and start re-using existing slots, collect
	 * previous results and delete the monitor before re-using the slot:
	 */
	if (m->valid) {
		finish_monitor(m);
	}

	init_monitor(m);

	egl->glBeginPerfMonitorAMD(m->id);
	m->active = true;
}

void end_perfcntrs(void)
{
	const struct egl *egl = perfcntr.egl;

	if (!egl) {
		return;
	}

	struct gl_monitor *m = &perfcntr.monitors[perfcntr.current_monitor];

	assert(m->valid);
	assert(m->active);

	/* end collection, but defer collecting results to avoid stall: */
	egl->glEndPerfMonitorAMD(m->id);
	m->active = false;

	/* move to next slot: */
	perfcntr.current_monitor =
		(perfcntr.current_monitor + 1) % ARRAY_SIZE(perfcntr.monitors);
}

/* collect any remaining perfcntr results.. this should be called
 * before computing the elapsed time (passed to dump_perfcntrs())
 * to ensured queued up draws which are monitored complete, ie.
 * so that elapsed time covers the entirety of the monitored
 * draws.
 */
void finish_perfcntrs(void)
{
	if (!perfcntr.egl)
		return;

	/* collect any remaining results, it really doesn't matter the order */
	for (unsigned i = 0; i < ARRAY_SIZE(perfcntr.monitors); i++) {
		struct gl_monitor *m = &perfcntr.monitors[i];
		if (m->valid) {
			finish_monitor(m);
		}
	}
}

void dump_perfcntrs(unsigned nframes, uint64_t elapsed_time_ns)
{
	if (!perfcntr.egl) {
		return;
	}

	/* print column headers: */
	printf("FPS");
	for (unsigned i = 0; i < perfcntr.num_counters; i++) {
		struct counter *c = &perfcntr.counters[i];

		printf(",%s", perfcntr.groups[c->gidx].counters[c->cidx].name);
	}
	printf("\n");

	/* print results: */
	double secs = elapsed_time_ns / (double)NSEC_PER_SEC;
	printf("%f", (double)nframes/secs);
	for (unsigned i = 0; i < perfcntr.num_counters; i++) {
		struct counter *c = &perfcntr.counters[i];

		GLuint counter_type =
			perfcntr.groups[c->gidx].counters[c->cidx].counter_type;
		switch (counter_type) {
		case GL_UNSIGNED_INT:
			printf(",%u", c->result.u32);
			break;
		case GL_FLOAT:
			printf(",%f", c->result.f);
			break;
		case GL_UNSIGNED_INT64_AMD:
			printf(",%"PRIu64, c->result.u64);
			break;
		case GL_PERCENTAGE_AMD:
		default:
			errx(-1, "TODO unhandled counter type: 0x%04x",
				counter_type);
			break;
		}
	}
	printf("\n");
}
