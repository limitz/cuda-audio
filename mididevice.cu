#include "mididevice.h"

static bool isValidMessage(const uint8_t* buffer, size_t len)
{
	if (!len) return false;
	switch (buffer[0] & 0xF0)
	{
	case 0x80:
	case 0x90:
	case 0xA0:
	case 0xB0:
		return (3 <= len);
	case 0xF0:
		return (0xF7 == buffer[len-1]);
	default:
		return true;
	}
}



void* MidiDevice::proc(void* context)
{
	auto self = reinterpret_cast<MidiDevice*>(context);
	self->_isRunning = true;

	int rc;
	uint8_t buffer[512];
	size_t len = 0;

	while (self->_isRunning)
	{
		assert(self->_isOpen);

		if (isValidMessage(buffer, len))
		{
			auto h = self->handler;
			if (h)
			{
				h->midiDeviceHandlerOnReceive(self, buffer, len);
			}
			//else
			{
				for (size_t i=0; i<len; i++)
				{
					std::cout << std::hex << (int)buffer[i] << " ";
				}
				std::cout << std::endl;
			}
			len = 0;
		}
		else
		{
			uint8_t byte;
			rc = snd_rawmidi_read(self->_in, &byte, 1);
			if (-EAGAIN == rc)
			{
				usleep(1000);
				continue;
			}
			if (0 > rc)
			{
				if (!self->_isRunning) return nullptr;
				assert(false || "Error reading from midi input");
			}
			if (byte & 0x80) self->_runningStatus = byte;
			else if (!len) buffer[len++] = self->_runningStatus;
			buffer[len++] = byte;
		}
	}
	self->_isRunning = false;

	return nullptr;
}

void MidiDevice::start()
{
	assert(!_isOpen);

	int rc;
	rc = snd_rawmidi_open(&_in, &_out, deviceId.c_str(), 0);
	assert(0 == rc);
	assert(&_in);
	assert(&_out);

	rc = snd_rawmidi_nonblock(_out, 1);
	assert(0 == rc);

	_isOpen = true;

	rc = pthread_create(&_thread, NULL, MidiDevice::proc, this);
	assert(0 == rc);
}

void MidiDevice::stop()
{
	assert(_isOpen);
	if (_isRunning)
	{
		_isRunning = false;
		pthread_join(_thread, 0);
	}

	if (_in) 
	{
		snd_rawmidi_close(_in);
		_in = 0;
	}

	if (_out)
	{
		snd_rawmidi_drain(_out);
		snd_rawmidi_close(_out);
		_out = 0;
	}
}

