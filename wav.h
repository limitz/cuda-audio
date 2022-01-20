#pragma once

#include "gpu.h"

class WavFile
{
public:
	std::string path;
	size_t numFrames;
	float2 *buffer;
	WavFile(const std::string& path);
	~WavFile() { if (buffer) cudaFree(buffer); }
};
