#include <cufft.h>
#include <iostream>

#include "gpu.h"
#include "audiodevice.h"
#include "mididevice.h"

__global__ static void f_makeTone(float* output, size_t channels, size_t samples, size_t sr, size_t t, float v)
{
	auto stride = gridDim * blockDim;
	auto offset = blockDim * blockIdx + threadIdx;

	assert(2 == channels);
	assert(1 == stride.y);
	assert(0 == offset.y);

	auto d = (float2*)output;
	for (auto s = offset.x; s < samples; s += stride.x)
	{
		float a1 = (t + s) % sr;
		float a2 = (t + s + (sr >> 1)) % sr;
		float b1 = v * fmaf(powf((sr-a1)/sr,10), fmodf(a1/40,2), -1);
		float b2 = v * fmaf(powf((sr-a2)/sr,10), fmodf(a2/40,2), -1);
		d[s] = {b1,b2};
	}
}

class MainHandler : public MidiDevice::Handler, public AudioDevice::Handler
{
public:
	MainHandler() :
		_fftSize(0),
		_a(nullptr), _afft(nullptr),
		_b(nullptr), _bfft(nullptr),
		_r(nullptr), _rfft(nullptr)
	{
	}

	void prepare(size_t size, size_t channels)
	{
		int rc;
		assert(!_fftSize);
		assert(size > 0);
		assert(channels > 0);

		_fftSize = size * channels * sizeof(cufftComplex);
		rc = cudaMalloc(&_a, _fftSize);
		assert(0 == rc);
		rc = cudaMalloc(&_b, _fftSize);
		assert(0 == rc);
		rc = cudaMalloc(&_r, _fftSize);
		assert(0 == rc);

		rc = cudaMalloc(&_afft, _fftSize);
		assert(0 == rc);
		rc = cudaMalloc(&_bfft, _fftSize);
		assert(0 == rc);
		rc = cudaMalloc(&_rfft, _fftSize);
		assert(0 == rc);
	}

	

protected:
	virtual void midiDeviceHandlerOnReceive(MidiDevice* sender, const uint8_t* buffer, size_t len) override
	{
		std::cout << "Received message" << std::endl;
	}

	virtual void audioDeviceHandlerOnOutputBuffer(AudioDevice* sender, float* buffer, size_t frames) override
	{
		static size_t t = 0;
		auto nc = sender->numChannels;
		auto sr = sender->sampleRate;
		f_makeTone <<< 2, 256, 0, 0 >>> (buffer, nc, frames, sr, t, 0.15f);
		cudaStreamSynchronize(0);
		t += frames;
	}

private:
	cufftComplex *_a, *_afft;
	cufftComplex *_b, *_bfft;
	cufftComplex *_r, *_rfft;

	size_t _fftSize;

};

int main()
{
	selectGpu();

	MainHandler handler;

	AudioDevice sound("default", &handler);
	sound.start();

	MidiDevice midi("hw:3,0,0");
	midi.start();

	std::cin.get();

	midi.stop();
	sound.stop();
	
	return 0;
}
