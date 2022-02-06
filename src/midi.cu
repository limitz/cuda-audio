#include "midi.h"

static bool isValidMessage(const uint8_t* buffer, size_t len)
{
	if (!len) return false;
	switch (buffer[0] & 0xF0)
	{
	case 0x80:
	case 0x90:
	case 0xA0:
	case 0xB0:
		return 3 == len;
	case 0xF0:
		return 0xF7 == buffer[len - 1];
	default:
		Log::warn("midi", "Unexpected midi byte: %02x", buffer[0]);
		assert(false && "Unexpected midi byte");
		return false;
	}
}

void* RawMidi::Device::proc(void* arg)
{
	auto self = static_cast<RawMidi::Device*>(arg);
	assert(self);

	int rc;
	auto buffer = (uint8_t*) alloca(256);
	size_t len = 0;

	self->isRunning = true;
	while (self->isRunning)
	{
		if (isValidMessage(buffer, len))
		{
			if (self->handler) self->handler->onMidiMessage(self, buffer, len);
			len = 0;
		}
		else
		{
			uint8_t byte;
			rc = snd_rawmidi_read(self->in, &byte, 1);
			if (-EAGAIN == rc)
			{
				usleep(1000);
				continue;
			}
			if (0 > rc)
			{
				if (!self->isRunning) return nullptr;
				assert(false && "Error reading from midi input");
			}
			if (byte & 0x80) self->runningStatus = byte;
			else if (!len) buffer[len++] = self->runningStatus;
			buffer[len++] = byte;
		}
	}
	return nullptr;
}

void RawMidi::Device::start()
{
	assert(!isOpen);

	int rc;
	rc = snd_rawmidi_open(&in, &out, id.c_str(), 0);
	assert(0 == rc);
	//assert(in);
	//assert(out);
	
	assert(in || out);

	if (out)
	{
		rc = snd_rawmidi_nonblock(out, 1);
		assert(0 == rc);
	}
	isOpen = true;

	if (in)
	{
		rc = pthread_create(&thread, NULL, RawMidi::Device::proc, this);
		assert(0 == rc);
	}
}

void RawMidi::Device::stop()
{
	assert(isOpen);
	if (isRunning)
	{
		isRunning = false;
		pthread_join(thread, 0);
	}

	if (in)
	{
		snd_rawmidi_close(in);
		in = nullptr;
	}
	if (out)
	{
		snd_rawmidi_drain(out);
		snd_rawmidi_close(out);
		out = nullptr;
	}
	isOpen = false;
}
