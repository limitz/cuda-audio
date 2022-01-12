#include <iostream>
#include "audiodevice.h"
#include "mididevice.h"

class MainHandler : public MidiDeviceHandler, public AudioDeviceHandler
{
protected:
	virtual void midiDeviceHandlerOnReceive(const uint8_t* buffer, size_t len) override
	{
		std::cout << "Received message" << std::endl;
	}

	virtual void audioDeviceHandlerOnOutputBuffer(float* buffer, size_t channels, size_t frames) override
	{
	}
};

int main()
{
	MainHandler handler;

	AudioDevice sound("default", &handler);
	sound.start();

	MidiDevice midi("hw:3,0,0");
	midi.start();

	std::cin.get();

	midi.stop();
	sound.stop();
	
	return 0;
}
