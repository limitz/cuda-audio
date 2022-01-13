#include "audiodevice.h"

#define TAG "AudioDevice"

static int xrunRecovery(snd_pcm_t* pcm, int err)
{
	int rc;
	switch(err)
	{
	case -EPIPE:
		Log::warn(TAG, "XRUN recovery: EPIPE");
		rc = snd_pcm_prepare(pcm);
		if (0 != rc) Log::error(TAG, "Unable to prepare sound device: %d", rc);
		return rc;

	case -ESTRPIPE:
		Log::warn("AudioDevice", "XRUN recovery: ESTRPIPE");
		while ((rc = snd_pcm_resume(pcm)) == -EAGAIN) usleep(100);
		if (rc < 0) 
		{
			rc = snd_pcm_prepare(pcm);
			if (0 != rc) Log::error(TAG, "Unable to prepare sound device: %d", rc);
		}
		return rc;

	default: 
		Log::error("AudioDevice", "XRUN not recoverable: %d", err);
		return err;
	}
}

static int waitForPoll(snd_pcm_t* pcm, struct pollfd *fd, unsigned int count)
{
	unsigned short revents;
	while (true)
	{
		poll(fd, count, -1);
		snd_pcm_poll_descriptors_revents(pcm, fd, count, &revents);
		if (revents & POLLERR)
		{
			Log::error(TAG, "Poll error");
			return -EIO;
		}
		if (revents & POLLOUT)
		{
			return 0;
		}
	}
}

void* AudioDevice::proc(void* context)
{
	int rc;
	AudioDevice* self = reinterpret_cast<AudioDevice*>(context);

	self->_isRunning = true;

	pthread_t thread = pthread_self();
	struct sched_param sched;
	sched.sched_priority = sched_get_priority_max(SCHED_FIFO);
	sched.sched_priority -= (1 + sched.sched_priority / 10);
	pthread_setschedparam(thread, SCHED_FIFO, &sched);
	int r = 0;
	pthread_getschedparam(thread, &r, &sched);
	//doesn't seem to work for now
	//assert(SCHED_FIFO == r);

	bool needsPoll = false;
	while (self->_isRunning)
	{
		size_t requestFrames = self->periodSize*2;

		assert(self->_isOpen);
		if (needsPoll)
		{
			rc = waitForPoll(self->_pcm, self->_poll.fd, self->_poll.count);
			assert(0 == rc);
		}

		auto h = self->handler;
		if (h)
		{
			h->audioDeviceHandlerOnOutputBuffer(self, self->_buffer, requestFrames);
		}

		snd_pcm_uframes_t written = 0;
		while (written < requestFrames)
		{	
			rc = snd_pcm_writei(
					self->_pcm, 
					self->_buffer + written * self->numChannels, 
					min(self->periodSize, requestFrames - written));
			if (rc < 0)
			{
				rc = xrunRecovery(self->_pcm, rc);
				assert(0 == rc);
				break;
			}
			needsPoll = true;
			written += rc;
			if (written >= requestFrames) break;
		
			rc = waitForPoll(self->_pcm, self->_poll.fd, self->_poll.count);
			assert(0 == rc);
		}
	}
	self->_isRunning = false;
	return nullptr;
}

void AudioDevice::start()
{
	int rc;
	assert(!_isOpen);

	rc = snd_pcm_open(&_pcm, deviceId.c_str(), SND_PCM_STREAM_PLAYBACK, 0);
	assert(0 == rc);

	rc = snd_pcm_nonblock(_pcm, 1);
	assert(0 == rc);

	snd_pcm_hw_params_t* hw;
	snd_pcm_hw_params_alloca(&hw);
	snd_pcm_hw_params_any(_pcm, hw);

	rc = snd_pcm_hw_params_set_rate_resample(_pcm, hw, 0);
	assert(0 == rc);

	rc = snd_pcm_hw_params_set_access(_pcm, hw, SND_PCM_ACCESS_RW_INTERLEAVED);
	assert(0 == rc);

	rc = snd_pcm_hw_params_set_format(_pcm, hw, SND_PCM_FORMAT_FLOAT_LE);
	assert(0 == rc);

	rc = snd_pcm_hw_params_set_channels(_pcm, hw, numChannels);
	assert(0 == rc);

	rc = snd_pcm_hw_params_set_rate(_pcm, hw, sampleRate, 0);
	assert(0 == rc);

	//rc = snd_pcm_hw_params_get_buffer_size_min(hw, &bufferSize);
	//assert(0 == rc);

	rc = snd_pcm_hw_params_set_buffer_size(_pcm, hw, bufferSize);
	assert(0 == rc);
	std::cout << bufferSize << std::endl;

	int dir = -1;
	//rc = snd_pcm_hw_params_get_period_size_min(hw, &periodSize, &dir);
	//assert(0 == rc);
	
	rc = snd_pcm_hw_params_set_period_size_near(_pcm, hw, &periodSize, &dir);
	assert(0 == rc);
	std::cout << periodSize << std::endl;

	rc = snd_pcm_hw_params(_pcm, hw);
	assert(0 == rc);

	snd_pcm_sw_params_t* sw;
	snd_pcm_sw_params_alloca(&sw);

	rc = snd_pcm_sw_params_current(_pcm, sw);
	assert(0 == rc);

	rc = snd_pcm_sw_params_set_start_threshold(_pcm, sw, (bufferSize / periodSize) * periodSize);
	assert(0 == rc);

	rc = snd_pcm_sw_params_set_avail_min(_pcm, sw, periodSize);
	assert(0 == rc);

	rc = snd_pcm_sw_params(_pcm, sw);
	assert(0 == rc);

	_poll.count = snd_pcm_poll_descriptors_count(_pcm);
	assert(_poll.count > 0);

	_poll.fd = (struct pollfd*)calloc(_poll.count, sizeof(struct pollfd));
	assert(_poll.fd);

	snd_pcm_poll_descriptors(_pcm, _poll.fd, _poll.count);

	rc = cudaHostAlloc(&_buffer, bufferSize * numChannels * sizeof(float), cudaHostAllocMapped);
	assert(rc == 0);
	assert(_buffer);

	_isOpen = true;

	rc = pthread_create(&_thread, NULL, AudioDevice::proc, this);
	assert(0 == rc);
}

void AudioDevice::stop()
{
	assert(_isOpen);
	
	if (_isRunning)
	{
		_isRunning = false;
		pthread_join(_thread, 0);
	}

	snd_pcm_drain(_pcm);
	snd_pcm_close(_pcm);
	free(_poll.fd);
	cudaFree(_buffer);

	_isOpen = false;
}


