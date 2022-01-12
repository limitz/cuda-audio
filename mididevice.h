#pragma once

#include <alsa/asoundlib.h>
#include "gpu.h"

class MidiDevice
{
public:
	class Handler
	{	
	protected:
		friend class MidiDevice;
		virtual void midiDeviceHandlerOnReceive(MidiDevice*, const uint8_t* message, size_t len) = 0;
	};

	const std::string deviceId;
	Handler* handler;

	MidiDevice(const std::string& deviceId, Handler* handler = nullptr) :
		deviceId(deviceId),
		handler(handler),
		_in(0), _out(0),
		_isOpen(false),
		_runningStatus(0),
		_thread(0)
	{
	}

	void start();
	void stop();
	void send(uint8_t* message, size_t len);

private:
	uint8_t _runningStatus;
	bool _isOpen, _isRunning;
	snd_rawmidi_t* _in;
	snd_rawmidi_t* _out;

	pthread_t _thread;

	static void* proc(void*);

};
