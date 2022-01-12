#pragma once

#include <alsa/asoundlib.h>
#include <pthread.h>
#include <string>
#include <cstdint>

class MidiDeviceHandler
{	
protected:
	virtual void midiDeviceHandlerOnReceive(const uint8_t* message, size_t len) = 0;
	friend class MidiDevice;
};

class MidiDevice
{
public:
	const std::string deviceId;
	MidiDeviceHandler* handler;

	MidiDevice(const std::string& deviceId, MidiDeviceHandler* handler = nullptr) :
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
