#pragma once
#include <cufft.h>
#include <map>

#include "gpu.h"
#include "wav.h"
#include "jackclient.h"

#ifndef CONV_FFTSIZE
#define CONV_FFTSIZE (512 * 256)
#endif

#ifndef CONV_GRIDSIZE
#define CONV_GRIDSIZE 256
#endif

#ifndef CONV_BLOCKSIZE
#define CONV_BLOCKSIZE 256
#endif

class Convolution : public JackClient
{
public:
	struct CC
	{
		uint8_t message;
		uint8_t select, predelay, dry, wet;
	} cc;

	Convolution(const std::string& name = "Conv", uint8_t ccMessage = 0xB0, uint8_t ccStart = 0x15, size_t fftSize = CONV_FFTSIZE);
	
	// TODO make destructor that destroys all buffers

	JackPort midiIn;
	JackPort input;
	JackPort left;
	JackPort right;

	virtual void onProcess(size_t nframes) override;
	virtual void onStart() override;
	inline double avgRuntime() const { return _nruns ? _runtime / _nruns : 0; }

	void prepare(size_t idx, const WavFile& wav, size_t nframes = 512);

private:
	cufftHandle _plan;
	cufftComplex* cin;
	cufftComplex* cinFFT;

	struct
	{
		cufftComplex* left, *right;
	} ir, irFFT, output, residual;

	std::map<size_t, cufftComplex*> _irBuffers;

	size_t _widx = 0; // index of IR wav file
	size_t _maxPredelay = 8192;
	float _predelay = 0.0f;
	float _wet = 1.0f;
	float _dry = 0.0f;

	size_t _interpolationSteps = 1000;
	double _runtime = 0;
	int _nruns = -10;
	size_t _delay = 1600;
	size_t _lp = 8;
	float _vol = 0.4f;
	size_t _fftSize;
	cudaStream_t _streams[4];
};

