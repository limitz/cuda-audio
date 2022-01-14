#pragma once
#include <alsa/asoundlib.h>
#include "gpu.h"

class AudioDevice
{
public:
	enum class Mode
	{
		output,
		input
	} mode;

	class Handler
	{
	protected:
		friend class AudioDevice;
		virtual void audioDeviceOnOutputBuffer(AudioDevice*, float* buffer, size_t frames) {};
		virtual void audioDeviceOnInputBuffer(AudioDevice*, float* buffer, size_t frames) {};
	};

	const std::string deviceId;
	Handler* handler;

	const int numChannels = 2;
	const int sampleRate = 48000;
	snd_pcm_uframes_t periodSize = 2048;
	snd_pcm_uframes_t bufferSize = 4096;

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
	static void* proc_output(void*);
	static void* proc_input(void*);
};
