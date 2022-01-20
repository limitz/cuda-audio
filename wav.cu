#include "wav.h"
#include <fstream>

__global__ static void f_wavConvert(float2* output, short2* input, size_t frames)
{
	auto stride = gridDim * blockDim;
	auto offset = blockDim * blockIdx + threadIdx;

	for (int s = offset.x; s < frames; s += stride.x)
	{
		short2 v = input[s];
		output[s] = make_float2(
			v.x / (float)65536,
			v.y / (float)65536);
	}
}

WavFile::WavFile(const std::string& path) : path(path)
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
	is.read((char*)&header, 8);

	char* format = (char*) alloca(4);
	is.read(format, 4);
	assert(!memcmp(format, "WAVE", 4));

	is.read((char*)&header, 8);
	assert(header.chunkSize >= sizeof(fmt_t));
	fmt_t* fmt = (fmt_t*) alloca(header.chunkSize);
	is.read((char*)fmt, header.chunkSize);

	Log::info("WAV", "Format: %d", fmt->audioFormat);
	Log::newline("Num Channels: %d", fmt->numChannels);
	Log::newline("Sample Rate: %d", fmt->sampleRate);
	Log::newline("Byte Rate: %d", fmt->byteRate);
	Log::newline("Block Align: %d", fmt->blockAlign);
	Log::newline("Bits per Sample: %d", fmt->bitsPerSample);

	is.read((char*)&header, 8);
	Log::info("WAV", "reading %d frames of audio (%0.2f s)", 
			header.chunkSize / fmt->blockAlign, 
			header.chunkSize / (float)fmt->byteRate);
	char* hostBuffer = new char[header.chunkSize];
	is.read(hostBuffer, header.chunkSize);

	char* devBuffer;
	int rc = cudaMalloc(&devBuffer, header.chunkSize);
	assert(cudaSuccess == rc);

	numFrames = header.chunkSize / (fmt->numChannels * (fmt->bitsPerSample >> 3));

	rc = cudaMalloc(&buffer, numFrames * sizeof(float2));
	assert(cudaSuccess == rc);

	rc = cudaMemcpy(devBuffer, hostBuffer, header.chunkSize, cudaMemcpyHostToDevice);
	assert(cudaSuccess == rc);

	assert(2 == fmt->numChannels);
	assert(4 == fmt->blockAlign);
	assert(16 == fmt->bitsPerSample);
	f_wavConvert <<< 16, 256, 0, 0 >>> ( buffer, (short2*)devBuffer, numFrames);

	delete[] hostBuffer;
	
	cudaStreamSynchronize(0);

	Log::info("WAV", "Ready. %d frames", numFrames);
}
