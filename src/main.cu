#include <cufft.h>
#include <ncurses.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include "gpu.h"
#include "wav.h"
#include "jackclient.h"
#include "conv.h"

#ifndef NUM_CONV_INSTANCES
#define NUM_CONV_INSTANCES 1 
#endif

int main()
{
	selectGpu();


	// Top row of my novation launchcontrol starts at 0x15
	uint8_t ccMessage = 0xB0;
	uint8_t ccStart = 0x15;

	Convolution* instances[NUM_CONV_INSTANCES] = { nullptr };
	for (auto i=0UL; i < NUM_CONV_INSTANCES; i++)
	{
		char* name = (char*)alloca(256);
		sprintf(name, "cudaconv_%lu",i+1);

		// There are 8 controls, let's assume simply that cc is contiguous
		// Other mappings would require changing Convolution::cc member
		auto c = instances[i] = new Convolution(name, ccMessage + i, ccStart);
	
		std::ifstream is("index.txt");
		std::string path;
		for (size_t idx = 0; std::getline(is, path); idx++)
		{
			WavFile w(path);
			c->prepare(idx, w);
		}
		c->start();

		// TODO get connections from settings
		// Connect inputs, assumed to be available
		jack_connect(c->handle, "system:capture_1", jack_port_name(c->capture[0]));
		jack_connect(c->handle, "system:capture_2", jack_port_name(c->capture[1]));
		
		// Connect to stereo output, assumed to be available
		jack_connect(c->handle, jack_port_name(c->playback[0]),  "system:playback_1");
		jack_connect(c->handle, jack_port_name(c->playback[1]), "system:playback_2");

		// Auto connect all MIDI ports
		#if 1
		auto midiports = jack_get_ports(c->handle, NULL, JACK_DEFAULT_MIDI_TYPE, JackPortIsOutput);
		for (auto midiport = midiports; *midiport; midiport++)
		{
			Log::info(__func__, "Found MIDI port: %s", *midiport);
			jack_connect(c->handle, *midiport, jack_port_name(c->midiIn));
		}
		jack_free(midiports);
		#endif

	}

	std::cin.get();

	for (auto i=0UL; i < NUM_CONV_INSTANCES; i++)
	{
		auto c = instances[i];
		if (c->isRunning()) c->stop();
		Log::info(c->name, "Average convolution runtime: %f", c->avgRuntime());
		delete c;
	}

	return 0;
}
