#pragma once

#include "gpu.h"
#include <jack/jack.h>
#include <jack/midiport.h>
#include <map>

typedef jack_port_t* JackPort;

class JackClient
{
public:
	const std::string name;
	jack_client_t* handle;
	size_t samplerate;

	std::map<std::string, JackPort> ports;

	JackClient(const std::string& name);
	virtual ~JackClient();

protected:
	JackPort addInput(const std::string& name, const std::string& type = JACK_DEFAULT_AUDIO_TYPE, size_t bufferSize = 0)
	{
		Log::info(name, "Registering %s input port: %s (buffer: %d)", 
				type.c_str(), name.c_str(), bufferSize);
		JackPort p = jack_port_register(handle, name.c_str(), type.c_str(), JackPortIsInput, bufferSize);
		assert(p);
		ports[name] = p;
		return p;
	}

	JackPort addOutput(const std::string& name, const std::string& type = JACK_DEFAULT_AUDIO_TYPE, size_t bufferSize = 0)
	{
		Log::info(name, "Registering %s output port: %s (buffer: %d)", 
				type.c_str(), name.c_str(), bufferSize);
		JackPort p = jack_port_register(handle, name.c_str(), type.c_str(), JackPortIsOutput, bufferSize);
		assert(p);
		ports[name] = p;
		return p;
	}

	inline void activate() 
	{
		jack_activate(handle);
		Log::info(name, "Activated.");
	}
	virtual void onProcess(size_t nframes) = 0;
	virtual void onShutdown() {};

private:
	static int processCallback(jack_nframes_t, void* arg);
	static void shutdownCallback(void* arg);
};
