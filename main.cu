#include <cufft.h>
#include <iostream>
#include <sstream>
#include <vector>
#include "gpu.h"
#include "wav.h"
#include "jackclient.h"
#include "conv.h"

#ifndef NUM_CONV_INSTANCES
#define NUM_CONV_INSTANCES 1 
#endif

// TODO: remove after refactoring, extern decl in conv.cu
WavFile* wav[38];

int main()
{
	selectGpu();

	wav[0] = new WavFile("ir/1/Block Inside.wav");
	wav[1] = new WavFile("ir/1/Bottle Hall.wav");
	wav[2] = new WavFile("ir/1/Cement Blocks 1.wav");
	wav[3] = new WavFile("ir/1/Cement Blocks 2.wav");
	wav[4] = new WavFile("ir/1/Chateau de Logne, Outside.wav");
	wav[5] = new WavFile("ir/1/Conic Long Echo Hall.wav");
	wav[6] = new WavFile("ir/1/Deep Space.wav");
	wav[7] = new WavFile("ir/1/Derlon Sanctuary.wav");
	wav[8] = new WavFile("ir/1/Direct Cabinet N1.wav");
	wav[9] = new WavFile("ir/1/Direct Cabinet N2.wav");
	wav[10] = new WavFile("ir/1/Direct Cabinet N3.wav");
	wav[11] = new WavFile("ir/1/Direct Cabinet N4.wav");
	wav[12] = new WavFile("ir/1/Five Columns.wav");
	wav[13] = new WavFile("ir/1/Five Columns Long.wav");
	wav[14] = new WavFile("ir/1/French 18th Century Salon.wav");
	wav[15] = new WavFile("ir/1/Going Home.wav");
	wav[16] = new WavFile("ir/1/Greek 7 Echo Hall.wav");
	wav[17] = new WavFile("ir/1/Highly Damped Large Room.wav");
	wav[18] = new WavFile("ir/1/In The Silo.wav");
	wav[19] = new WavFile("ir/1/In The Silo Revised.wav");
	wav[20] = new WavFile("ir/1/Large Bottle Hall.wav");
	wav[21] = new WavFile("ir/1/Large Long Echo Hall.wav");
	wav[22] = new WavFile("ir/1/Large Wide Echo Hall.wav");
	wav[23] = new WavFile("ir/1/Masonic Lodge.wav");
	wav[24] = new WavFile("ir/1/Musikvereinsaal.wav");
	wav[25] = new WavFile("ir/1/Narrow Bumpy Space.wav");
	wav[26] = new WavFile("ir/1/Nice Drum Room.wav");
	wav[27] = new WavFile("ir/1/On a Star.wav");
	wav[28] = new WavFile("ir/1/Parking Garage.wav");
	wav[29] = new WavFile("ir/1/Rays.wav");
	wav[30] = new WavFile("ir/1/Right Glass Triangle.wav");
	wav[31] = new WavFile("ir/1/Ruby Room.wav");
	wav[32] = new WavFile("ir/1/Scala Milan Opera Hall.wav");
	wav[33] = new WavFile("ir/1/Small Drum Room.wav");
	wav[34] = new WavFile("ir/1/Small Prehistoric Cave.wav");
	wav[35] = new WavFile("ir/1/St Nicolaes Church.wav");
	wav[36] = new WavFile("ir/1/Trig Room.wav");
	wav[37] = new WavFile("ir/1/Vocal Duo.wav");

	// Top row of my novation launchcontrol starts at 0x15
	uint8_t startCC = 0x15;

	Convolution* instances[NUM_CONV_INSTANCES];
	for (auto i=0UL; i < NUM_CONV_INSTANCES; i++)
	{
		char* name = (char*)alloca(256);
		sprintf(name, "cudaconv_%lu",i+1);

		// There are 4 controls, let's assume simply that cc is contiguous
		// Other mappings would require changing Convolution::cc member
		auto c = instances[i] = new Convolution(name, startCC + 4 * i);
		c->start();

		// Auto connect to capture_<i+1>, there are nicer ways to do this but connecting ports
		// is going to be dealt with later in different code.
		#if 1
		sprintf(name, "system:capture_%d", i+1);
		jack_connect(c->handle, name, jack_port_name(c->input));
		#else
		jack_connect(c->handle, "system:capture_1", jack_port_name(c->input));
		#endif
		
		// Connect to stereo output, assumed to be available
		jack_connect(c->handle, jack_port_name(c->left),  "system:playback_1");
		jack_connect(c->handle, jack_port_name(c->right), "system:playback_2");

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

	Log::info(__func__, "%sPress <enter> to quit...", escapeRgb(255,0,128).c_str());
	std::cin.get();
	
	for (auto i=0UL; i < NUM_CONV_INSTANCES; i++)
	{
		auto c = instances[i];
		if (c->isRunning()) c->stop();
		Log::info(c->name, "Average convolution runtime: %f", c->avgRuntime());
		delete c;
	}


	for (auto i=0UL; i<sizeof(wav)/sizeof(*wav); i++) delete wav[i];
	return 0;
}
