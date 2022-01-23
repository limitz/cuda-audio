#include "jackclient.h"


int JackClient::processCallback(jack_nframes_t nframes, void* arg)
{
	assert(arg);
	auto jc = static_cast<JackClient*>(arg);
	jc->onProcess(nframes);

	return 0;
}

void JackClient::shutdownCallback(void* arg)
{
	auto jc = static_cast<JackClient*>(arg);
	Log::warn(jc->name, "Shutting down...");
	jc->onShutdown();
}

JackClient::JackClient(const std::string& name) : name(name), _isRunning(false)
{
}

void JackClient::start()
{
	jack_status_t status;

	Log::info(name, "Starting JACK plugin");

	handle = jack_client_open(name.c_str(), JackNoStartServer, &status, NULL);
	assert(handle);
	assert(!(status & JackNameNotUnique));

	Log::info(name, "Client opened");

	jack_set_process_callback(handle, processCallback, this);
	jack_on_shutdown(handle, shutdownCallback, this);

	samplerate = jack_get_sample_rate(handle);
	Log::info(name, "Samplerate: %d", samplerate);

	_isRunning = true;
	onStart();
}

void JackClient::stop()
{
	assert(handle);
	assert(_isRunning);
	onStop();

	jack_client_close(handle);
	_isRunning = false;
	usleep(500000);
}

JackClient::~JackClient()
{
}


