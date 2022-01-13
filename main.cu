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
		float b1 = v * fmaf(powf((sr-a1)/sr,10), fmodf(a1/40,2), -1);
		output[s] = {b1,0};
	}
}

__global__ static void f_makeImpulseResponse(cufftComplex* output, size_t samples, size_t sr, size_t t, float v)
{
	auto stride = gridDim * blockDim;
	auto offset = blockDim * blockIdx + threadIdx;

	assert(1 == stride.y);
	assert(0 == offset.y);

	for (auto s = offset.x; s < samples; s += stride.x)
	{
		output[s] ={s == 1 ? 1.0f : 0, 0};
	}
}

__global__ static void f_pointwiseAdd(cufftComplex* r, const cufftComplex* a, const cufftComplex* b, size_t size)
{
	auto stride = gridDim * blockDim;
	auto offset = blockDim * blockIdx + threadIdx;

	for (auto s = offset.x; s < size; s += stride.x)
	{
		auto va = a[s];
		auto vb = b[s];
		r[s] = va;// + vb;
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
		_b(nullptr), _bfft(nullptr),
		_r(nullptr), _rfft(nullptr)
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
		rc = cudaMalloc(&_b, fftSize);
		assert(0 == rc);
		rc = cudaMalloc(&_r, fftSize);
		assert(0 == rc);

		rc = cudaMalloc(&_afft, fftSize);
		assert(0 == rc);
		rc = cudaMalloc(&_bfft, fftSize);
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
		std::cout << "Received message" << std::endl;
	}

	virtual void audioDeviceHandlerOnOutputBuffer(AudioDevice* sender, float* buffer, size_t frames) override
	{
		int rc;
		static size_t t = 0;
		auto nc = sender->numChannels;
		auto sr = sender->sampleRate;
		cudaMemset(_a, 0, _fftSize * sizeof(cufftComplex));
		f_makeTone <<< 2, 256, 0, _streams[0] >>> (_a, frames, sr, t, 0.15f);
		f_makeImpulseResponse <<< 4, 256, 0, _streams[1] >>> (_b, 4096, sr, 0, 1.0f);
		cudaStreamSynchronize(_streams[0]);
		cudaStreamSynchronize(_streams[1]);
		cufftSetStream(_plan, _streams[0]);

		rc = cufftExecC2C(_plan, _a, _afft, CUFFT_FORWARD);
		assert(cudaSuccess == rc);
		rc = cufftExecC2C(_plan, _b, _bfft, CUFFT_FORWARD);
		assert(cudaSuccess == rc);
		
		f_pointwiseMultiply <<< 8, 256, 0, _streams[0] >>> (_rfft, _afft, _bfft, _fftSize);
		
		rc = cufftExecC2C(_plan, _rfft, _r, CUFFT_INVERSE);
		assert(cudaSuccess == rc);

		f_pointwiseAdd <<< 4, 256, 0, _streams[0] >>> (_a, _r, _residual, _fftSize);

		rc = cudaMemcpyAsync(_residual, _r+frames, frames * sizeof(cufftComplex), cudaMemcpyDeviceToDevice, _streams[0]);
		assert(cudaSuccess == rc);
		rc = cudaMemcpyAsync(buffer, _a, frames*sizeof(cufftComplex), cudaMemcpyDeviceToHost, _streams[0]);
		assert(cudaSuccess == rc);

		rc = cudaStreamSynchronize(_streams[0]);
		assert(cudaSuccess == rc);
		t += frames;
	}

private:
	cufftHandle _plan;
	cufftComplex *_a, *_afft;
	cufftComplex *_b, *_bfft;
	cufftComplex *_r, *_rfft;
	cufftComplex *_residual;

	cudaStream_t _streams[4];
	size_t _fftSize;

};

int main()
{
	selectGpu();

	MainHandler handler;
	handler.prepare(4096, 2);

	AudioDevice sound("default", &handler);
	sound.start();

	MidiDevice midi("hw:3,0,0");
	midi.start();

	std::cin.get();

	midi.stop();
	sound.stop();
	
	return 0;
}
