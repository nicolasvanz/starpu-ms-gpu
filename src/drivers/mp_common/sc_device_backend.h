/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#ifndef __SC_DEVICE_BACKEND_H__
#define __SC_DEVICE_BACKEND_H__

#include <stddef.h>
#include <drivers/mp_common/mp_common.h>   /* enum _starpu_mp_impl_kind */

#pragma GCC visibility push(hidden)
#ifdef STARPU_USE_MP

/* Pluggable accelerator backend for the server-client sink path. One backend
 * per accelerator tool (CUDA). CPU has NO backend: it is the baseline host
 * path (_starpu_sc_backend_for returns NULL for CPU). */
struct _starpu_sc_device_backend
{
	const char *name;                       /* "cuda", "opencl", "hip" */
	enum _starpu_mp_impl_kind impl_kind;

	int   (*device_count)(void);            /* # of such devices visible on this sink */
	void *(*lookup)(const char *symbol);    /* resolve a kernel symbol on the sink */
	void  (*set_device)(int devid);         /* select the device before a kernel launch */
	void  (*synchronize)(void);             /* wait for the kernel after launch */
	void *(*alloc)(size_t size);            /* managed alloc; NULL => caller uses host malloc */
	int   (*owns_ptr)(void *ptr);           /* does this backend own ptr? (free dispatch) */
	void  (*free_ptr)(void *ptr);
};

/* Backend for an accelerator impl_kind, or NULL for CPU / a backend not built in. */
const struct _starpu_sc_device_backend *_starpu_sc_backend_for(enum _starpu_mp_impl_kind impl_kind);

#endif /* STARPU_USE_MP */
#pragma GCC visibility pop
#endif /* __SC_DEVICE_BACKEND_H__ */
