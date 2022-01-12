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
		float a = (t + s) % sr;
		float b = v * fmaf(sinpif(a/8000), sinpif(a/40), 1);
		d[s] = {-b,b};
	}
}

class MainHandler : public MidiDevice::Handler, public AudioDevice::Handler
{
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
