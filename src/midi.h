#pragma once
#include <alsa/asoundlib.h>
#include <pthread.h>
#include <string>
#include <vector>
#include <cstdint>
#include <cassert>

#include "log.h"

class RawMidi
{
public:
	class Device;

	class MessageHandler
	{
	public:
		virtual void onMidiMessage(const Device* sender, const uint8_t *buffer, size_t len) = 0;
	};

	class Device
	{
	public:
		Device(const std::string& id) :
			id(id), in(nullptr), out(nullptr), thread(0), 
			isOpen(false), isRunning(false), runningStatus(0),
			handler(nullptr)
		{
		}

		virtual ~Device() {}
		void start();
		void stop();
		void send(uint8_t* data, size_t len);

		MessageHandler* handler;
		std::string id;
		snd_rawmidi_t* in;
		snd_rawmidi_t* out;
		pthread_t thread;
		bool isOpen, isRunning;
		uint8_t runningStatus;
	
		static void* proc(void*);
	};
	
	std::vector<Device> connected;

	void connectAllDevices();
};
