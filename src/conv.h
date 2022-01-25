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

#ifndef CONV_MAX_ISTEPS
#define CONV_MAX_ISTEPS 512
#endif

#ifndef CONV_MAX_PREDELAY
#define CONV_MAX_PREDELAY 8192
#endif

class Convolution : public JackClient
{
public:
	struct CC
	{
		uint8_t message;
		uint8_t select, predelay, dry, wet, isteps, panDry, panWet1, panWet2;
		struct
		{
			size_t select = 0;   // [0-size]
			size_t predelay = 0; // [0-8192]
			size_t isteps = 100; // [0-512]
			size_t vsteps = 0;
			float dry = 0.5f; // [0,1]
			float wet = 0.5f; // [0,1]
			float panDry  = 0.0f; // [-1,1]
			float panWet1 = 0.0f; // [-1,1]
			float panWet2 = 0.0f; // [-1,1]
		} value;
	} cc1, cc2;

	Convolution(const std::string& name = "Conv", uint8_t ccMessage = 0xB0, uint8_t ccStart = 0x15, size_t fftSize = CONV_FFTSIZE);
	
	// TODO make destructor that destroys all buffers

	JackPort midiIn;
	JackPort capture[2];
	JackPort playback[2];

	virtual void onProcess(size_t nframes) override;
	virtual void onStart() override;
	inline double avgRuntime() const { return _nruns ? _runtime / _nruns : 0; }

	void prepare(size_t idx, const WavFile& wav, size_t nframes = 512);

private:
	cufftHandle _plan;
	cufftComplex* cinFFT;
	cufftComplex* cin;

	struct
	{
		cufftComplex* left, *right;
	} ir, irFFT, output, residual;

	std::map<size_t, cufftComplex*> _irBuffers;

	double _runtime = 0;
	int _nruns = -10;
	size_t _delay = 1600;
	size_t _lp = 8;
	float _vol = 0.4f;
	size_t _fftSize;
	cudaStream_t _streams[4];
};

