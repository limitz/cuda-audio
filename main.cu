#include <cufft.h>
#include <iostream>

#include "gpu.h"
#include "audiodevice.h"
#include "mididevice.h"

__global__ static void f_makeTone(cufftComplex* output, size_t samples, size_t sr, size_t t, float v)
{
	auto stride = gridDim * blockDim;
	auto offset = blockDim * blockIdx + threadIdx;

	assert(1 == stride.y);
	assert(0 == offset.y);

	for (auto s = offset.x; s < samples; s += stride.x)
	{
		float a1 = (t + s) % sr;
		float b1 = v * fmaf(powf((sr-a1)/sr,20), fmodf(a1/(50+s/200),2), -1);
		output[s] = {b1,0};
	}
}

__global__ static void f_makeImpulseResponse(cufftComplex* output, size_t samples, size_t sr, size_t t, size_t dt, size_t lp,float v)
{
	auto stride = gridDim * blockDim;
	auto offset = blockDim * blockIdx + threadIdx;

	assert(1 == stride.y);
	assert(0 == offset.y);

	for (auto s = offset.x; s < samples; s += stride.x)
	{
		output[s] = {((s + dt) % t) < lp ? 1.0f / (lp+lp*s/t) : 0.0f, 0.0f};
	}
}

__global__ static void f_pointwiseAdd(cufftComplex* r, const cufftComplex* s1, const cufftComplex* s2, const cufftComplex* b, size_t size)
{
	auto stride = gridDim * blockDim;
	auto offset = blockDim * blockIdx + threadIdx;

	for (auto s = offset.x; s < size; s += stride.x)
	{
		auto vs1 = s1[s];
		auto vs2 = s2[s];
		auto res = b[s];
		r[s] = {vs1.x + res.x, vs2.x + res.y};
	}
}

__global__ static void f_pointwiseMultiply(cufftComplex* r, const cufftComplex* a, const cufftComplex* b, size_t size)
{
	auto stride = gridDim * blockDim;
	auto offset = blockDim * blockIdx + threadIdx;

	for (auto s = offset.x; s < size; s += stride.x)
	{
		auto va = a[s];
		auto vb = b[s];
		auto re = va.x * vb.x - va.y * vb.y;
		auto im = (va.x + va.y) * (vb.x + vb.y) - re;
		r[s] = make_float2(re, im) / size;
	}
}

class MainHandler : public MidiDevice::Handler, public AudioDevice::Handler
{
public:
	MainHandler() :
		_fftSize(0),
		_a(nullptr), _afft(nullptr),
		_b1(nullptr), _b1fft(nullptr),
		_b2(nullptr), _b2fft(nullptr),
		_r1(nullptr), _r2(nullptr), _rfft(nullptr)
	{
		int rc;
		for (auto i = 0; i<4; i++) 
		{
			rc = cudaStreamCreateWithFlags(&_streams[i], cudaStreamNonBlocking);
			assert(0 == rc);
		}
	}

	void prepare(size_t size, size_t channels)
	{
		int rc;
		assert(!_fftSize);
		assert(size > 0);
		assert(channels > 0);

		auto fftSize = size * sizeof(cufftComplex) * 2;
		rc = cudaMalloc(&_a, fftSize);
		assert(0 == rc);
		rc = cudaMalloc(&_b1, fftSize);
		assert(0 == rc);
		rc = cudaMalloc(&_b2, fftSize);
		assert(0 == rc);
		rc = cudaMalloc(&_r1, fftSize);
		assert(0 == rc);
		rc = cudaMalloc(&_r2, fftSize);
		assert(0 == rc);

		rc = cudaMalloc(&_afft, fftSize);
		assert(0 == rc);
		rc = cudaMalloc(&_b1fft, fftSize);
		assert(0 == rc);
		rc = cudaMalloc(&_b2fft, fftSize);
		assert(0 == rc);
		rc = cudaMalloc(&_rfft, fftSize);
		assert(0 == rc);
		rc = cudaMalloc(&_residual, fftSize);
		assert(0 == rc);

		cudaMemset(_residual, 0, fftSize);
		_fftSize = size;
		int n[] = { (int)_fftSize };
		int iembed[] = { (int)_fftSize };
		int oembed[] = { (int)_fftSize };
		int istride = 1;
		int ostride = 1;
		int batch = 1;

		rc = cufftPlanMany(&_plan, 1, n, 
				iembed, istride, _fftSize, 
				oembed, ostride, _fftSize,
				CUFFT_C2C, batch);
		assert(0 == rc);
	}	

protected:
	virtual void midiDeviceHandlerOnReceive(MidiDevice* sender, const uint8_t* buffer, size_t len) override
	{
		if (buffer[0] == 0xB0 && buffer[1] == 0x15) _delay = 200 + 16000 * buffer[2] / 0x80;
		if (buffer[0] == 0xB0 && buffer[1] == 0x16) _lp = 1 + buffer[2];
		if (buffer[0] == 0xB0 && buffer[1] == 0x17) _vol = buffer[2];
	}

	virtual void audioDeviceHandlerOnOutputBuffer(AudioDevice* sender, float* buffer, size_t frames) override
	{
		int rc;
		
		cudaEvent_t started, stopped;
		cudaEventCreate(&started);
		cudaEventCreate(&stopped);

		static size_t t = 0;
		auto nc = sender->numChannels;
		auto sr = sender->sampleRate;
		cudaEventRecord(started, _streams[0]);
		cudaMemset(_a, 0, _fftSize * sizeof(cufftComplex));
		f_makeTone <<< 2, 256, 0, _streams[0] >>> (_a, frames, sr, t, _vol / 256.0f);
		f_makeImpulseResponse <<< 32, 256, 0, _streams[0] >>> (_b1, _fftSize, sr, _delay, 0, _lp, 1.0f);
		f_makeImpulseResponse <<< 32, 256, 0, _streams[0] >>> (_b2, _fftSize, sr, _delay, _delay/2, _lp, 1.0f);
		cufftSetStream(_plan, _streams[0]);

		rc = cufftExecC2C(_plan, _a, _afft, CUFFT_FORWARD);
		assert(cudaSuccess == rc);
		
		cudaStreamSynchronize(_streams[1]);
		rc = cufftExecC2C(_plan, _b1, _b1fft, CUFFT_FORWARD);
		assert(cudaSuccess == rc);
		f_pointwiseMultiply <<< 64, 256, 0, _streams[0] >>> (_rfft, _afft, _b1fft, _fftSize);
		rc = cufftExecC2C(_plan, _rfft, _r1, CUFFT_INVERSE);
		assert(cudaSuccess == rc);

		cudaStreamSynchronize(_streams[2]);
		rc = cufftExecC2C(_plan, _b2, _b2fft, CUFFT_FORWARD);
		assert(cudaSuccess == rc);
		f_pointwiseMultiply <<< 64, 256, 0, _streams[0] >>> (_rfft, _afft, _b2fft, _fftSize);
		rc = cufftExecC2C(_plan, _rfft, _r2, CUFFT_INVERSE);
		assert(cudaSuccess == rc);
		
		f_pointwiseAdd <<< 4, 256, 0, _streams[0] >>> (_a, _r1, _r2, _residual, _fftSize - frames);
		rc = cudaMemcpyAsync(_residual, _a+frames, (_fftSize - frames) * sizeof(cufftComplex), 
				cudaMemcpyDeviceToDevice, _streams[0]);
		assert(cudaSuccess == rc);
		rc = cudaMemcpyAsync(buffer, _a, frames*sizeof(cufftComplex), cudaMemcpyDeviceToHost, _streams[0]);
		assert(cudaSuccess == rc);

		cudaEventRecord(stopped, _streams[0]);

		rc = cudaStreamSynchronize(_streams[0]);
		assert(cudaSuccess == rc);
		t += frames;
	
		float elapsed;
		rc = cudaEventElapsedTime(&elapsed, started, stopped);
		assert(cudaSuccess == rc);

		std::cout << elapsed << std::endl;
	}

private:
	size_t _delay = 1600;
	size_t _lp = 8;
	size_t _vol = 0x30;

	cufftHandle _plan;
	cufftComplex *_a, *_afft;
	cufftComplex *_b1, *_b1fft;
	cufftComplex *_b2, *_b2fft;
	cufftComplex *_r1, *_r2, *_rfft;
	cufftComplex *_residual;

	cudaStream_t _streams[4];
	size_t _fftSize;

};

int main()
{
	selectGpu();

	MainHandler handler;
	handler.prepare(48000, 2);

	AudioDevice sound("default", &handler);
	sound.start();

	MidiDevice midi("hw:3,0,0", &handler);
	midi.start();

	std::cin.get();

	midi.stop();
	sound.stop();
	
	return 0;
}
