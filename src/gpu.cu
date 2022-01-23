#include "gpu.h"

#define TAG "GPU"

static int smToCores(int major, int minor)
{
	switch ((major << 4) | minor)
	{
		case (9999 << 4 | 9999):
			return 1;
		case 0x30:
		case 0x32:
		case 0x35:
		case 0x37:
			return 192;
		case 0x50:
		case 0x52:
		case 0x53:
			return 128;
		case 0x60:
			return 64;
		case 0x61:
		case 0x62:
			return 128;
		case 0x70:
		case 0x72:
		case 0x75:
			return 64;
		case 0x80:
			return 64;
		case 0x86:
			return 128;
		default:
			return 0;
	};
}

void selectGpu()
{
	int rc;
	int maxId = -1;
	uint16_t maxScore = 0;
	int count = 0;
	cudaDeviceProp prop;

	rc = cudaGetDeviceCount(&count);
	assert(cudaSuccess == rc);
	assert(count > 0);


	for (int id = 0; id < count; id++)
	{
		rc = cudaGetDeviceProperties(&prop, id);
		assert(cudaSuccess == rc);
		
		if (prop.computeMode == cudaComputeModeProhibited) 
		{
			Log::warn(TAG, "GPU %d: (%s) is prohibited", id, prop.name);
			continue;
		}
		int sm_per_multiproc = smToCores(prop.major, prop.minor);
		
		Log::info(TAG, "GPU %d", id);
		Log::newline(ESC(1) "%s" ESC(0), prop.name);
		Log::newline("Compute capability: " ESC(1) "%d.%d" ESC(0), prop.major, prop.minor);
		Log::newline("Multiprocessors:    " ESC(1) "%d" ESC(0), prop.multiProcessorCount);
		Log::newline("SMs per processor:  " ESC(1) "%d" ESC(0), sm_per_multiproc);
		Log::newline("Clock rate:         " ESC(1) "%d" ESC(0), prop.clockRate);
		Log::newline();

		uint64_t score =(uint64_t) prop.multiProcessorCount * sm_per_multiproc * prop.clockRate;
		if (score > maxScore) 
		{
			maxId = id;
			maxScore = score;
		}
	}


	assert(maxId >= 0);

	rc = cudaSetDevice(maxId);
	assert(cudaSuccess == rc);

	rc = cudaGetDeviceProperties(&prop, maxId);
	assert(cudaSuccess == rc);

	Log::info(__func__, ESC(32;1) "Selected GPU %d: \"%s\" with compute capability %d.%d", 
		maxId, prop.name, prop.major, prop.minor);
}
