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

#include <starpu.h>
#include <common/config.h>
#include <drivers/mp_common/sc_device_backend.h>

#ifdef STARPU_USE_MP

#ifdef STARPU_USE_CUDA
#include <dlfcn.h>
#include <starpu_cuda.h>
#include <cuda_runtime_api.h>

static int cuda_device_count(void)
{
	int n = 0;
	if (cudaGetDeviceCount(&n) != cudaSuccess)
		n = 0;
	return n;
}

static void *cuda_lookup(const char *symbol)
{
#ifdef RTLD_DEFAULT
	return dlsym(RTLD_DEFAULT, symbol);
#else
	void *dl_handle = dlopen(NULL, RTLD_NOW);
	return dlsym(dl_handle, symbol);
#endif
}

static void cuda_set_device(int devid)
{
	STARPU_ASSERT_MSG(devid >= 0, "missing sink CUDA device id for CUDA task");
	starpu_cuda_set_device((unsigned) devid);
}

static void cuda_synchronize(void)
{
	cudaError_t cures = cudaDeviceSynchronize();
	STARPU_ASSERT_MSG(cures == cudaSuccess, "CUDA task execution failed on sink");
}

static void *cuda_alloc(size_t size)
{
	void *addr = NULL;
	if (cudaMallocManaged(&addr, size, cudaMemAttachGlobal) != cudaSuccess)
		addr = NULL;
	return addr;
}

static int cuda_owns_ptr(void *ptr)
{
	struct cudaPointerAttributes attributes;
	cudaError_t cures = cudaPointerGetAttributes(&attributes, ptr);
#if CUDART_VERSION >= 10000
	int owns = (cures == cudaSuccess && attributes.type != cudaMemoryTypeUnregistered);
#else
	int owns = (cures == cudaSuccess && attributes.memoryType != cudaMemoryTypeHost);
#endif
	(void) cudaGetLastError();
	return owns;
}

static void cuda_free_ptr(void *ptr)
{
	cudaFree(ptr);
}

static const struct _starpu_sc_device_backend cuda_backend =
{
	.name         = "cuda",
	.impl_kind    = STARPU_MP_IMPL_CUDA,
	.device_count = cuda_device_count,
	.lookup       = cuda_lookup,
	.set_device   = cuda_set_device,
	.synchronize  = cuda_synchronize,
	.alloc        = cuda_alloc,
	.owns_ptr     = cuda_owns_ptr,
	.free_ptr     = cuda_free_ptr,
};
#endif /* STARPU_USE_CUDA */

const struct _starpu_sc_device_backend *_starpu_sc_backend_for(enum _starpu_mp_impl_kind impl_kind)
{
#ifdef STARPU_USE_CUDA
	if (impl_kind == STARPU_MP_IMPL_CUDA)
		return &cuda_backend;
#endif
	(void) impl_kind;
	return NULL;   /* CPU baseline, or backend not built in */
}

#endif /* STARPU_USE_MP */
