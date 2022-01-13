#pragma once

#include "gpu.h"

class WavFile
{
public:
	size_t numFrames;
	float2 *buffer;
	WavFile(const std::string& path);
	~WavFile() { if (buffer) cudaFree(buffer); }
};
