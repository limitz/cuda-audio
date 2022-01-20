#pragma once

#include <cufft.h>
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
	Convolution(const std::string& name = "Conv", size_t fftSize = CONV_FFTSIZE);

	JackPort midiIn;
	JackPort input;
	JackPort left;
	JackPort right;

	virtual void onProcess(size_t nframes) override;
	inline double avgRuntime() const { return _nruns ? _runtime / _nruns : 0; }

	void loadIR(size_t idx, const WavFile& wav);

private:
	cufftHandle _plan;
	
	cufftComplex* cin;
	cufftComplex* cinFFT;

	struct
	{
		cufftComplex* left, *right;
	} ir, irFFT, output, residual;

	size_t _widx = 0; // index of IR wav file
	size_t _predelay = 0;
	float _wet = 1.0;
	float _dry = 0.0;

	double _runtime = 0;
	int _nruns = -10;
	size_t _delay = 1600;
	size_t _lp = 8;
	float _vol = 0.4f;
	size_t _fftSize;
	cudaStream_t _streams[4];
};

