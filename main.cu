#include <cufft.h>
#include <iostream>

#include "gpu.h"
#include "wav.h"
#include "jackclient.h"

static WavFile* wav[38];

__global__ static void f_makeTone(cufftComplex* output, size_t samples, size_t sr, size_t t, float v)
{
	auto stride = gridDim * blockDim;
	auto offset = blockDim * blockIdx + threadIdx;

	for (auto s = offset.x; s < samples; s += stride.x)
	{
		float a1 = (t + s) % sr;
		float b1 = v * fmaf(powf(0.1f+0.9f*(sr-a1)/sr,10), min(a1/200,1.0f) * sinpif(a1/sr * 190) + cospif(a1/sr * 256), 0);
		output[s] = {b1,0};
	}
}

__global__ static void f_deinterleaveIR(cufftComplex* L, cufftComplex* R, float2* ir, size_t n)
{
	auto stride = gridDim * blockDim;
	auto offset = blockDim * blockIdx + threadIdx;

	for (auto s = offset.x; s < n; s += stride.x)
	{
		auto v = ir[s];
		L[s] = {v.x, 0};
		R[s] = {v.y, 0};
	}
}

__global__ static void f_pointwiseAdd(cufftComplex* r, const cufftComplex* a, size_t n)
{
	auto stride = gridDim * blockDim;
	auto offset = blockDim * blockIdx + threadIdx;

	for (auto s = offset.x; s < n; s += stride.x)
	{
		auto v = r[s] + clamp(a[s], -1, 1);
		r[s] = v;
	}
}

__global__ static void f_scale(cufftComplex* r, float scale, size_t n)
{
	auto stride = gridDim * blockDim;
	auto offset = blockDim * blockIdx + threadIdx;

	for (auto s = offset.x; s < n; s += stride.x)
	{
		r[s] = clamp(r[s] * scale, -1, 1);
	}
}
__global__ static void f_pointwiseMultiply(cufftComplex* r, const cufftComplex* ir, const cufftComplex* a, size_t n)
{
	auto stride = gridDim * blockDim;
	auto offset = blockDim * blockIdx + threadIdx;

	for (auto s = offset.x; s < n; s += stride.x)
	{
		auto va = a[s];
		auto vb = ir[s];
		auto re = va.x * vb.x - va.y * vb.y;
		auto im = (va.x + va.y) * (vb.x + vb.y) - re;
		r[s] = make_float2(re, im);
	}
}

class Convolution : public JackClient
{
public:
	Convolution(size_t fftSize = 512 * 256) : JackClient("Conv"),
		_fftSize(fftSize)
	{
		int rc;
		for (auto i = 0; i<4; i++) 
		{
			rc = cudaStreamCreateWithFlags(&_streams[i], cudaStreamNonBlocking);
			assert(0 == rc);
		}


		cufftComplex** cc[] = {
			&cin, &cinFFT, 
			&ir.left, &ir.right,
			&irFFT.left, &irFFT.right,
			&output.left, &output.right,
			&residual.left, &residual.right
		};
		for (auto i = 0; i < sizeof(cc) / sizeof(*cc); i++)
		{
			rc = cudaMalloc(cc[i], fftSize * sizeof(cufftComplex));
			assert(cudaSuccess == rc);
		}

		int n[] = { fftSize };
		int inembed[] = { (int)fftSize };
		int istride   = 1;
		int idist     = (int)fftSize;
		int onembed[] = { (int)fftSize };
		int ostride   = 1;
		int odist     = (int)fftSize;
		int batchSize = 1;

		rc = cufftPlanMany(&_plan, 1, n, 
				inembed, istride, idist,
				onembed, ostride, odist,
				CUFFT_C2C, batchSize);
		assert(0 == rc);

		activate();
		midiIn = addInput("input.midi", JACK_DEFAULT_MIDI_TYPE);
		input = addInput("input.mono");
		left = addOutput("output.left");
		right = addOutput("output.right");
	}

	cufftHandle _plan;
	cufftComplex* cin;
	cufftComplex* cinFFT;

	struct
	{
		cufftComplex* left, *right;
	} ir, irFFT, output, residual;

	JackPort midiIn;
	JackPort input;
	JackPort left;
	JackPort right;

	void onProcess(size_t nframes)
	{
		int rc;
		auto in = jack_port_get_buffer(input, nframes);
		auto L = jack_port_get_buffer(left, nframes);
		auto R = jack_port_get_buffer(right, nframes);


		auto midi = jack_port_get_buffer(midiIn, nframes);
		auto nevts = jack_midi_get_event_count(midi);
		for (auto i=0;i<nevts; i++)
		{
			jack_midi_event_t evt;
			rc = jack_midi_event_get(&evt, midi, i);
			assert(0 == rc);
		
			for (auto c=0; c<evt.size; c++)
			{
				//std::cout << std::hex << (int)evt.buffer[c] << " ";
			}
			//std::cout << std::endl;
			if ((evt.buffer[0] & 0xF0) == 0x90)
			{
				_widx = (_widx + 1) % 38;
				_uploadIR = true;
			}
		}
		


		cudaEvent_t started, stopped;
		cudaEventCreate(&started);
		cudaEventCreate(&stopped);
		cudaEventRecord(started, _streams[0]);
		cufftSetStream(_plan, _streams[0]);
	
		if (_uploadIR)
		{
			// move impulse response to irFFT.left , irFFT.right
			f_deinterleaveIR <<< 32, 256, 0, _streams[1] >>> (
				ir.left, ir.right,
				wav[_widx]->buffer, 
				min(wav[_widx]->numFrames, _fftSize - nframes));
		}
		// copy input to device
		rc = cudaMemcpy2DAsync(
				cin,  sizeof(cufftComplex), 
				in,   sizeof(float), 
				sizeof(float), nframes,
				cudaMemcpyHostToDevice, _streams[0]);
		assert(cudaSuccess == rc);
		
		// get FFT of input
		rc = cufftExecC2C(_plan, cin, cinFFT, CUFFT_FORWARD);
		assert(cudaSuccess == rc);

		if (_uploadIR)
		{
			// await deinterleaveIR
			rc = cudaStreamSynchronize(_streams[1]);
			assert(cudaSuccess == rc);
		
			// inplace transform irFFT.left and irFFT.right
			rc = cufftExecC2C(_plan, ir.left, irFFT.left, CUFFT_FORWARD);
			assert(cudaSuccess == rc);
			rc = cufftExecC2C(_plan, ir.right, irFFT.right, CUFFT_FORWARD);
			assert(cudaSuccess == rc);
		}

		// multiply ir with input
		f_pointwiseMultiply <<< 256, 256, 0, _streams[0] >>> (output.left, irFFT.left, cinFFT, _fftSize);
		f_pointwiseMultiply <<< 256, 256, 0, _streams[0] >>> (output.right, irFFT.right, cinFFT, _fftSize);

		// take the inverse FFT of the output
		rc = cufftExecC2C(_plan, output.left, output.left, CUFFT_INVERSE);
		assert(cudaSuccess == rc);
		rc = cufftExecC2C(_plan, output.right, output.right, CUFFT_INVERSE);
		assert(cudaSuccess == rc);
		f_scale <<< 256, 256, 0, _streams[0] >>> (output.right, _vol * 1.0f/_fftSize, _fftSize);
		f_scale <<< 256, 256, 0, _streams[0] >>> (output.left,  _vol * 1.0f/_fftSize, _fftSize);
		
		// Add the residual
		f_pointwiseAdd <<< 256, 256, 0, _streams[0] >>> (output.left, residual.left, _fftSize - nframes);
		f_pointwiseAdd <<< 256, 256, 0, _streams[0] >>> (output.right, residual.right, _fftSize - nframes);
		
		// Copy output to host
		rc = cudaMemcpy2DAsync(L, sizeof(float), output.left, sizeof(cufftComplex),
				sizeof(float), nframes, cudaMemcpyDeviceToHost, _streams[0]);
		assert(cudaSuccess == rc);

		rc = cudaMemcpy2DAsync(R, sizeof(float), output.right, sizeof(cufftComplex),
				sizeof(float), nframes, cudaMemcpyDeviceToHost, _streams[0]);
		assert(cudaSuccess == rc);
		
		// Copy the residual for next cycle
		rc = cudaMemcpyAsync(
				residual.left, 
				output.left + nframes, 
				(_fftSize - nframes) * sizeof(cufftComplex), 
				cudaMemcpyDeviceToDevice, _streams[0]);
		assert(cudaSuccess == rc);
		rc = cudaMemcpyAsync(
				residual.right, 
				output.right + nframes, 
				(_fftSize - nframes) * sizeof(cufftComplex), 
				cudaMemcpyDeviceToDevice, _streams[0]);
		assert(cudaSuccess == rc);

		// Done
		cudaEventRecord(stopped, _streams[0]);
		rc = cudaStreamSynchronize(_streams[0]);
		assert(cudaSuccess == rc);

		float elapsed;
		rc = cudaEventElapsedTime(&elapsed, started, stopped);
		assert(cudaSuccess == rc);

		_runtime += elapsed;
		_nruns++;
		_uploadIR = false;
		//memcpy(L, in, nframes * sizeof(jack_default_audio_sample_t));
		//memcpy(R, in, nframes * sizeof(jack_default_audio_sample_t));
	}

	void onShutdown()
	{
	}

	double avgRuntime() const { return _nruns ? _runtime / _nruns : 0; }

private:
	double _runtime = 0;
	size_t _nruns = 0;
	size_t _delay = 1600;
	size_t _lp = 8;
	float _vol = 0.4f;
	size_t _widx = 0;
	bool _uploadIR = true;
	size_t _fftSize;
	cudaStream_t _streams[4];
};

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

	Convolution c;

	jack_connect(c.handle, "system:capture_1", jack_port_name(c.input));
	jack_connect(c.handle, jack_port_name(c.left),  "system:playback_1");
	jack_connect(c.handle, jack_port_name(c.right), "system:playback_2");

	/*
	auto midiports = jack_get_ports(c.handle, NULL, JACK_DEFAULT_MIDI_TYPE, JackPortIsOutput);
	while (*midiports)
	{
		jack_connect(c.handle, *midiports, jack_port_name(c.midiIn));
		midiports++;
	}
	jack_free(midiports);
	*/
	std::cin.get();

	Log::info(__func__, "Average convolution runtime: %f", c.avgRuntime());


	for (int i=0; i< 8; i++) delete wav[i];
	return 0;
}
