#include "wav.h"
#include <fstream>

__global__ static void f_wavConvert(float2* output, char* input, size_t frames, size_t numChannels, size_t bitsPerSample)
{
	auto stride = gridDim * blockDim;
	auto offset = blockDim * blockIdx + threadIdx;

	for (int s = 0; s < frames; s+=stride.x)
	{
		float r[2] = {0,0};
		for (int c = 0; c < numChannels; c++)
		{
			auto bps = bitsPerSample >> 3;
			auto idx = (s/2 + c) * bps * numChannels;
			uint32_t v = 0;
			for (int i=0; i<bps; i++)
			{
				v += ((uint32_t)input[idx + i]) << (i << 3);
			}
			r[c] = v;
		}
		output[s] = { r[0], r[1] };
	}
}

WavFile::WavFile(const std::string& path)
{
	struct hdr_t
	{
		uint32_t chunkId;
		uint32_t chunkSize;
	} header;

	struct fmt_t
	{
		uint16_t audioFormat;
		uint16_t numChannels;
		uint32_t sampleRate;
		uint32_t byteRate;
		uint16_t blockAlign;
		uint16_t bitsPerSample;
	};

	std::ifstream is = std::ifstream(path, std::ifstream::binary);
	is >> header.chunkId >> header.chunkSize;
	assert(header.chunkSize == 4);

	char* format = (char*) alloca(header.chunkSize);
	is.read(format, header.chunkSize);
	assert(!memcmp(format, "WAVE", 4));

	is >> header.chunkId >> header.chunkSize;
	assert(header.chunkSize >= sizeof(fmt_t));
	fmt_t* fmt = (fmt_t*) alloca(header.chunkSize);
	is.read((char*)fmt, header.chunkSize);

	is >> header.chunkId >> header.chunkSize;

	char* hostBuffer = new char[header.chunkSize];
	is.read(hostBuffer, header.chunkSize);

	char* devBuffer;
	int rc = cudaMalloc(&devBuffer, header.chunkSize);
	assert(cudaSuccess == rc);

	size_t frames = header.chunkSize / (fmt->numChannels * (fmt->bitsPerSample >> 3));

	rc = cudaMalloc(&buffer, frames * sizeof(float2));
	assert(cudaSuccess == rc);

	rc = cudaMemcpy(devBuffer, hostBuffer, header.chunkSize, cudaMemcpyHostToDevice);
	assert(cudaSuccess == rc);

	f_wavConvert <<< 16, 256, 0, 0 >>> ( buffer, devBuffer, frames, fmt->numChannels, fmt->bitsPerSample);

	delete[] hostBuffer;
	
	cudaStreamSynchronize(0);

}
