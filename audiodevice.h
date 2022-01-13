#pragma once
#include <alsa/asoundlib.h>
#include "gpu.h"

class AudioDevice
{
public:
	class Handler
	{
	protected:
		friend class AudioDevice;
		virtual void audioDeviceHandlerOnOutputBuffer(AudioDevice*, float* buffer, size_t frames) = 0;
	};

	const std::string deviceId;
	Handler* handler;

	const int numChannels = 2;
	const int sampleRate = 48000;
	snd_pcm_uframes_t periodSize = 1024*2;
	snd_pcm_uframes_t bufferSize = 2048*4;

	AudioDevice(const std::string& deviceId, Handler* handler = nullptr) : 
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
