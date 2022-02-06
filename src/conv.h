#pragma once
#include <cufft.h>
#include <map>

#include "gpu.h"
#include "wav.h"
#include "jackclient.h"
#include "midi.h"

#ifndef CONV_FFTSIZE
#define CONV_FFTSIZE (512 * 256)
#endif

#ifndef CONV_GRIDSIZE
#define CONV_GRIDSIZE 256
#endif

#ifndef CONV_BLOCKSIZE
#define CONV_BLOCKSIZE 256
#endif

#ifndef CONV_MAX_SPEED
#define CONV_MAX_SPEED 512
#endif

#ifndef CONV_MAX_PREDELAY
#define CONV_MAX_PREDELAY 8192
#endif

class Convolution : public JackClient, public RawMidi::MessageHandler
{
public:
	struct CC
	{
		RawMidi::Device* device;
		uint8_t message;
		uint8_t select, predelay, dry, wet, speed, panDry, panWet, level;
		struct
		{
			size_t select = 0;   // [0-size]
			size_t predelay = 0; // [0-8192]
			size_t speed = 100; // [0-512]
			size_t vsteps = 0;
			float dry = 0.5f; // [0,1]
			float wet = 0.5f; // [0,1]
			float panDry  = 0.0f; // [-1,1]
			float panWet = 0.0f; // [-1,1]
			float level = 1.0f; // [0,1]
		} value;
	} cc[2];

	Convolution(const std::string& name = "Conv", uint8_t ccMessage = 0xB0, uint8_t ccStart = 0x15, size_t fftSize = CONV_FFTSIZE);
	~Convolution() {}
	// TODO make destructor that destroys all buffers

	JackPort midiIn;
	JackPort capture[2];
	JackPort playback[2];

	virtual void onProcess(size_t nframes) override;
	virtual void onStart() override;
	inline double avgRuntime() const { return _nruns ? _runtime / _nruns : 0; }

	void prepare(size_t idx, const WavFile& wav, size_t nframes = 512);

	virtual void onMidiMessage(const RawMidi::Device* sender, const uint8_t *buffer, size_t len) override;

private:
	cufftHandle _plan;
	cufftComplex* cinFFT;
	cufftComplex *cin, *cin1, *cin2;

	struct
	{
		cufftComplex* left, *right;
	} ir, irFFT1, irFFT2, output, residual;

	std::map<size_t, cufftComplex*> _irBuffers;

	double _runtime = 0;
	int _nruns = -10;
	size_t _delay = 1600;
	size_t _lp = 8;
	float _vol = 0.4f;
	size_t _fftSize;
	cudaStream_t _streams[4];
};

