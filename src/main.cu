#include <cufft.h>
#include <ncurses.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <map>
#include "gpu.h"
#include "wav.h"
#include "jackclient.h"
#include "conv.h"
#include "settings.h"

#ifndef NUM_CONV_INSTANCES
#define NUM_CONV_INSTANCES 1
#endif

int main(int argc, char** argv)
{
	selectGpu();

	Settings settings;
	settings.open("settings.txt");

	auto numInstances = settings.u32("conv.count");
	assert(0 == (numInstances % 2) && "conv.count must be a multiple of 2");
	numInstances /= 2;

	std::map<std::string, RawMidi::Device*> midiDevices;

	Convolution** instances = new Convolution*[numInstances];
	for (auto n=0UL; n < numInstances; n++)
	{
		auto name = std::string("cudaconv_") + char('0' + n + 1);
		auto c = instances[n] = new Convolution(name);
		for (int i=0; i < 2; i++)
		{
			int idx = n * 2 + i;
			
			auto deviceId = settings.str("conv[%d].cc.device", idx);
			if (deviceId.size())
			{
				auto& mappedDevice = midiDevices[deviceId];
				if (!mappedDevice) mappedDevice = new RawMidi::Device(deviceId);
				c->cc[i].device = mappedDevice;
				// OK, so this needs some work
				// Only allows the same controller per pair of convolutions.
				c->cc[i].device->handler = c;
			}
			c->cc[i].message   = settings.u8("conv[%d].cc.message", idx);
			c->cc[i].select    = settings.u8("conv[%d].cc.select", idx);
			c->cc[i].predelay  = settings.u8("conv[%d].cc.predelay", idx);
			c->cc[i].dry       = settings.u8("conv[%d].cc.dry", idx);
			c->cc[i].wet       = settings.u8("conv[%d].cc.wet", idx);
			c->cc[i].speed     = settings.u8("conv[%d].cc.speed", idx);
			c->cc[i].panDry    = settings.u8("conv[%d].cc.panDry", idx);
			c->cc[i].panWet    = settings.u8("conv[%d].cc.panWet", idx);
			c->cc[i].level     = settings.u8("conv[%d].cc.level", idx);
			c->cc[i].value.select    = settings.u32("conv[%d].value.select", idx);
			c->cc[i].value.predelay  = settings.u32("conv[%d].value.predelay", idx);
			c->cc[i].value.dry       = settings.f32("conv[%d].value.dry", idx);
			c->cc[i].value.wet       = settings.f32("conv[%d].value.wet", idx);
			c->cc[i].value.speed     = settings.u32("conv[%d].value.speed", idx);
			c->cc[i].value.panDry    = settings.f32("conv[%d].value.panDry", idx);
			c->cc[i].value.panWet    = settings.f32("conv[%d].value.panWet", idx);
			c->cc[i].value.level     = settings.f32("conv[%d].value.level", idx);
	
			auto index = settings.str("conv[%d].index", idx);

			std::ifstream is(index);
			std::string path;
			for (size_t j = 0; std::getline(is, path); j++)
			{
				WavFile w(path);
				c->prepare(j, w);
			}
		}
		c->start();
		for (int i=0; i < 2; i++)
		{
			int idx = n * 2 + i;
			auto inputPort = settings.str("conv[%d].input", idx);
			auto outputPort = settings.str("conv[%d].output", idx);
			jack_connect(c->handle, inputPort.c_str(), jack_port_name(c->capture[i]));
			jack_connect(c->handle, jack_port_name(c->playback[i]), outputPort.c_str());
			auto d = c->cc[i].device;
			if (d && !d->isOpen) d->start();
		}
	}

	std::cin.get();

	for (auto i=0UL; i < numInstances; i++)
	{
		auto c = instances[i];
		for (auto n=0UL; n<2; n++)
		{
			auto d = c->cc[i].device;
			if (d && d->isOpen) d->stop();
		}
		if (c->isRunning()) c->stop();
		Log::info(c->name, "Average convolution runtime: %f", c->avgRuntime());
		delete c;
	}

	// TODO:
	// delete all midi device pointers in midiDevices map

	delete[] instances;

	return 0;
}
