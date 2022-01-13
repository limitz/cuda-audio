#pragma once

#include "gpu.h"

class WavFile
{
public:
	float2 *buffer;
	WavFile(const std::string& path);
	~WavFile() { if (buffer) cudaFree(buffer); }
};
