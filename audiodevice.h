#pragma once

#include <cuda_runtime.h>
#include <alsa/asoundlib.h>
#include <pthread.h>
#include <string>

class AudioDeviceHandler
{
protected:
	virtual void audioDeviceHandlerOnOutputBuffer(float* buffer, size_t channels, size_t frames) = 0;
	friend class AudioDevice;
};

class AudioDevice
{
public:
	const std::string deviceId;
	AudioDeviceHandler* handler;

	const int numChannels = 2;
	const int sampleRate = 48000;
	snd_pcm_uframes_t periodSize = 256;
	snd_pcm_uframes_t bufferSize = 512;

	AudioDevice(const std::string& deviceId, AudioDeviceHandler* handler = nullptr) : 
		deviceId(deviceId),
		handler(handler),
		_isOpen(false), 
		_isRunning(false), 
		_pcm(nullptr), 
		_buffer(nullptr), 
		_thread(0) 
	{}

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
