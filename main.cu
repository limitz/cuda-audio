#include <iostream>
#include "sounddevice.h"

int main()
{
	SoundDevice sound;
	sound.start();

	std::cin.get();

	sound.stop();
	
	return 0;
}
