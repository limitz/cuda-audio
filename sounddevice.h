#pragma once

#include <cuda_runtime.h>
#include <alsa/asoundlib.h>
#include <pthread.h>

class SoundDevice
{
public:
	const int numChannels = 2;
	const int sampleRate = 48000;
	snd_pcm_uframes_t periodSize = 256;
	snd_pcm_uframes_t bufferSize = 512;

	SoundDevice() : _isOpen(false), _isRunning(false), _pcm(nullptr), _buffer(nullptr), _thread(0) {}

	void start();
	void stop();

private:
	bool _isOpen, _isRunning;
	snd_pcm_t* _pcm;
	struct
	{
		struct pollfd* fd;
		int count;
	} _poll;

	pthread_t _thread;

	float *_buffer;
	static void* proc(void*);
};
