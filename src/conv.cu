#include "conv.h"

__global__ static void f_interpolate(
		cufftComplex* dst, cufftComplex* a, cufftComplex* b, 
		size_t fftSize, size_t steps, float wet)
{
	auto stride = gridDim * blockDim;
	auto offset = blockDim * blockIdx + threadIdx;

	for (int s = offset.x; s < fftSize; s += stride.x)
	{
		auto va = a[s];
		auto vb = b[s] * wet;
		auto vd = (vb - va) / (steps + 5); // just add a little to wet changes
		auto vv = va + vd;
		dst[s] = vv;
	}
}

__device__ inline cufftComplex conjugate(cufftComplex v) { return { v.x, -v.y }; }
__device__ inline cufftComplex timesj(cufftComplex v) { return { -v.y, v.x }; }

__global__ static void f_unpackC22R(cufftComplex* L, cufftComplex* R, cufftComplex* src, size_t fftSize)
{
	auto stride = gridDim * blockDim;
	auto offset = blockDim * blockIdx + threadIdx;

	assert(1 == __popcll(fftSize));
	auto m = fftSize - 1;

	for (auto s = offset.x; s < fftSize/2; s += stride.x)
	{
		auto idxa = s;
		auto idxb = (fftSize - s) & m;

		auto va = src[idxa];
		auto vb = conjugate(src[idxb]);
		auto la = 0.5f * (va + vb);
		auto lb = timesj(-0.5f * (va - vb));

		L[idxa] = la;
		R[idxa] = lb;
		L[idxb] = conjugate(la);
		R[idxb] = conjugate(lb);
	}
}

__global__ static void f_pointwiseAdd(cufftComplex* r, const cufftComplex* a, const cufftComplex* b, size_t n, size_t predelay)
{
	auto stride = gridDim * blockDim;
	auto offset = blockDim * blockIdx + threadIdx;

	for (auto s = offset.x; s < n; s += stride.x)
	{
		auto v = a[s];
		if (s >= predelay) v += b[s-predelay];
		r[s] = clamp(v, -1.0f, 1.0f);
	}
}

__global__ static void f_pointwiseMultiplyAndScale(cufftComplex* r, const cufftComplex* ir, const cufftComplex* a, size_t n, float scale)
{
	auto stride = gridDim * blockDim;
	auto offset = blockDim * blockIdx + threadIdx;

	for (auto s = offset.x; s < n; s += stride.x)
	{
		auto va = a[s];
		auto vb = ir[s];
		auto re = va.x * vb.x - va.y * vb.y;
		auto im = (va.x + va.y) * (vb.x + vb.y) - re;
		r[s] = make_float2(re, im) * scale;
	}
}

Convolution::Convolution(const std::string& name, uint8_t ccMessage, uint8_t ccStart, size_t fftSize) : 
	JackClient(name),
	_fftSize(fftSize),
	midiIn(nullptr),
	left(nullptr),
	right(nullptr),
	input(nullptr)
{
	int rc;
	for (auto i = 0; i<4; i++) 
	{
		rc = cudaStreamCreateWithFlags(&_streams[i], cudaStreamNonBlocking);
		assert(0 == rc);
	}


	cufftComplex** pcc[] = {
		&cin, &cinFFT, 
		&ir.left, &ir.right,
		&irFFT.left, &irFFT.right,
	};

	for (auto i = 0UL; i < sizeof(pcc) / sizeof(*pcc); i++)
	{
		rc = cudaMalloc(pcc[i], fftSize * sizeof(cufftComplex));
		assert(cudaSuccess == rc);
	}
		
	// TODO make this just float
	rc = cudaMalloc(&residual.left, (fftSize + _maxPredelay) * sizeof(cufftComplex));
	assert(cudaSuccess == rc);

	rc = cudaMalloc(&residual.right, (fftSize + _maxPredelay) * sizeof(cufftComplex));
	assert(cudaSuccess == rc);
	
	rc = cudaMalloc(&output.left, (fftSize + _maxPredelay) * sizeof(cufftComplex));
	assert(cudaSuccess == rc);

	rc = cudaMalloc(&output.right, (fftSize + _maxPredelay) * sizeof(cufftComplex));
	assert(cudaSuccess == rc);

	int n[] = { (int)fftSize };
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

	cc.message = ccMessage;
	cc.select = ccStart;
	cc.predelay = ccStart + 1;
	cc.dry = ccStart + 2;
	cc.wet = ccStart + 3;
	cc.isteps = ccStart + 4;
}

void Convolution::onStart()
{
	activate();
	midiIn = addInput("input.midi", JACK_DEFAULT_MIDI_TYPE);
	left = addOutput("output.left");
	right = addOutput("output.right");
	input = addInput("input.mono");
}

// TODO, make thread safe
void Convolution::prepare(size_t idx, const WavFile& wav, size_t nframes)
{
	int rc;
	
	cudaStream_t stream = _streams[0];
	cufftSetStream(_plan, stream);

	auto buf = _irBuffers[idx];
	if (buf) cudaFree(buf);

	// TODO pack real fft into half size buffer
	rc = cudaMalloc(&buf, sizeof(cufftComplex) * (_fftSize << 1));
	assert(cudaSuccess == rc);
	
	auto n = min(wav.numFrames, _fftSize - nframes);
	rc = cudaMemcpyAsync(buf, wav.buffer, sizeof(cufftComplex) * n, cudaMemcpyDeviceToDevice, stream);
	assert(cudaSuccess == rc);
	
	rc = cufftExecC2C(_plan, buf, buf, CUFFT_FORWARD);
	assert(cudaSuccess == rc);
	
	f_unpackC22R <<< CONV_GRIDSIZE, CONV_BLOCKSIZE, 0, stream >>> (buf, buf+_fftSize, buf,  _fftSize);

	cudaStreamSynchronize(stream);
	_irBuffers[idx] = buf;
}

void Convolution::onProcess(size_t nframes)
{
	int rc;

	auto in = input ? jack_port_get_buffer(input, nframes) : nullptr;
	auto L = left ? jack_port_get_buffer(left, nframes) : nullptr;
	auto R = right ? jack_port_get_buffer(right, nframes) : nullptr;
	auto midi = midiIn ? jack_port_get_buffer(midiIn, nframes) : nullptr;

	if (!in || !L || !R || !midi) return;

	auto nevts = jack_midi_get_event_count(midi);
	for (auto i=0UL;i<nevts; i++)
	{
		jack_midi_event_t evt;
		rc = jack_midi_event_get(&evt, midi, i);
		assert(0 == rc);
	
#if 0
		for (auto c=0; c<evt.size; c++) std::cout << std::hex << (int)evt.buffer[c] << " ";
		std::cout << std::endl;
#endif

		if (evt.buffer[0] == cc.message)
		{
			if (evt.buffer[1] == cc.select)
			{
				_widx = evt.buffer[2] * _irBuffers.size() / 0x80;
				_interpolationSteps = _maxInterpolationSteps;
				//std::cout << wav[_widx]->path.c_str() << std::endl;
			}
			else if (evt.buffer[1] == cc.predelay)
			{
				_predelay = evt.buffer[2] / 127.0f;
			}
			else if (evt.buffer[1] == cc.dry)
			{
				_dry = evt.buffer[2] / 127.0f;
			}
			else if (evt.buffer[1] == cc.wet)
			{
				_wet = evt.buffer[2] / 127.0f;
			}
			else if (evt.buffer[1] == cc.isteps)
			{
				_maxInterpolationSteps = evt.buffer[2] * 1000 / 0x80 + 1;
				if (_maxInterpolationSteps > _interpolationSteps)
				{
					_interpolationSteps = _maxInterpolationSteps;
				}
			}
		}
	}
	
	cudaEvent_t started, stopped;
	cudaEventCreate(&started);
	cudaEventCreate(&stopped);
	cudaEventRecord(started, _streams[0]);
	cufftSetStream(_plan, _streams[0]);


	// interpolate to IR FFT
	rc = cudaMemcpyAsync(ir.left, _irBuffers[_widx], sizeof(cufftComplex) * _fftSize, 
			cudaMemcpyDeviceToDevice, _streams[0]);
	assert(cudaSuccess == rc);
	f_interpolate <<< CONV_GRIDSIZE, CONV_BLOCKSIZE, 0, _streams[0] >>> (
			irFFT.left, irFFT.left, ir.left, _fftSize, _interpolationSteps, _wet);
	rc = cudaMemcpyAsync(ir.right, _irBuffers[_widx]+_fftSize, sizeof(cufftComplex) * _fftSize, 
			cudaMemcpyDeviceToDevice, _streams[0]);
	assert(cudaSuccess == rc);
	f_interpolate <<< CONV_GRIDSIZE, CONV_BLOCKSIZE, 0, _streams[0] >>> (
			irFFT.right, irFFT.right, ir.right, _fftSize, _interpolationSteps, _wet);
	if (_interpolationSteps > 1) _interpolationSteps--;

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

	// multiply ir with input
	f_pointwiseMultiplyAndScale <<< CONV_GRIDSIZE, CONV_BLOCKSIZE, 0, _streams[0] >>> (output.left, irFFT.left, cinFFT, _fftSize, 1.0f/_fftSize);
	f_pointwiseMultiplyAndScale <<< CONV_GRIDSIZE, CONV_BLOCKSIZE, 0, _streams[0] >>> (output.right, irFFT.right, cinFFT, _fftSize, 1.0f/_fftSize);

	auto tmp = ir;
	// take the inverse FFT of the output
	rc = cufftExecC2C(_plan, output.left, tmp.left, CUFFT_INVERSE);
	assert(cudaSuccess == rc);
	rc = cufftExecC2C(_plan, output.right, tmp.right, CUFFT_INVERSE);
	assert(cudaSuccess == rc);
		
	// Add the residual
	f_pointwiseAdd <<< CONV_GRIDSIZE, CONV_BLOCKSIZE, 0, _streams[0] >>> (output.left, residual.left, tmp.left, _fftSize, (size_t)(_predelay * _maxPredelay));
	f_pointwiseAdd <<< CONV_GRIDSIZE, CONV_BLOCKSIZE, 0, _streams[0] >>> (output.right, residual.right, tmp.right, _fftSize, (size_t)(_predelay * _maxPredelay));
	
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
			(_fftSize + _maxPredelay - nframes) * sizeof(cufftComplex), 
			cudaMemcpyDeviceToDevice, _streams[0]);
	assert(cudaSuccess == rc);
	rc = cudaMemcpyAsync(
			residual.right, 
			output.right + nframes, 
			(_fftSize + _maxPredelay - nframes) * sizeof(cufftComplex), 
			cudaMemcpyDeviceToDevice, _streams[0]);
	assert(cudaSuccess == rc);

	// Done
	cudaEventRecord(stopped, _streams[0]);
	cudaEventSynchronize(stopped);

	float elapsed;
	rc = cudaEventElapsedTime(&elapsed, started, stopped);
	assert(cudaSuccess == rc);

	// initialized nruns to negative value to discard first couple of runs
	if (++_nruns > 0) _runtime += elapsed;
	
	//memcpy(L, in, nframes * sizeof(jack_default_audio_sample_t));
	//memcpy(R, in, nframes * sizeof(jack_default_audio_sample_t));
}

