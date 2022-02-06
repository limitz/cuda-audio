#include "conv.h"

#ifndef CONV_INTERPOLATE
#define CONV_INTERPOLATE 1
#endif

#ifndef CONV_LOWPASS
#define CONV_LOWPASS 1
#endif

__device__ inline cufftComplex conjugate(cufftComplex v) { return { v.x, -v.y }; }
__device__ inline cufftComplex timesj(cufftComplex v) { return { -v.y, v.x }; }

#if CONV_INTERPOLATE
__global__ static void f_interpolate(
		cufftComplex* dst, const cufftComplex* a, const cufftComplex* b, 
		size_t fftSize, size_t steps, float wet)
{
	auto stride = gridDim * blockDim;
	auto offset = blockDim * blockIdx + threadIdx;

	//auto ns = 1.0f / (steps + 5);
	for (int s = offset.x; s < fftSize/2; s += stride.x)
	{
		auto va = a[s];
		auto vb = b[s] * wet;
		auto vd = (vb - va) / (steps + 5); //* ns;
		auto vv = va + vd;
		dst[s] = vv;
		if (s) dst[fftSize - s] = conjugate(vv);
	}
}
#endif

__global__ static void f_pack2R2C(cufftComplex* d, float* L, float* R, size_t n)
{
	auto stride = gridDim * blockDim;
	auto offset = blockDim * blockIdx + threadIdx;


	for (auto s = offset.x; s < n; s += stride.x)
	{
		d[s] = { L[s], R[s] };
	}
}

__global__ static void f_unpackC22R(cufftComplex* L, cufftComplex* R, const cufftComplex* src, size_t fftSize)
{
	auto stride = gridDim * blockDim;
	auto offset = blockDim * blockIdx + threadIdx;

	//assert(1 == __popcll(fftSize));
	//auto m = fftSize - 1;

	for (auto s = offset.x; s < fftSize/2; s += stride.x)
	{
		auto idxa = s;
		auto idxb = (fftSize - s);

		auto va = src[idxa];
		auto vb = s ? conjugate(src[idxb]) : va;
		auto la = 0.5f * (va + vb);
		auto lb = timesj(-0.5f * (va - vb));

		L[idxa] = la;
		R[idxa] = lb;
		if (s)
		{
			L[idxb] = conjugate(la);
			R[idxb] = conjugate(lb);
		}
	}
}


__global__ static void f_lowpass(cufftComplex* r, const cufftComplex* a, size_t fftSize)
{
	auto stride = gridDim * blockDim;
	auto offset = blockDim * blockIdx + threadIdx;

	for (auto s = offset.x; s < fftSize/2; s += stride.x)
	{
		auto va = a[s] * cbrt(0.54 - 0.46 * cospif((2.0f * s) / fftSize));
		r[s] = va;
		if (s) r[fftSize - s] = conjugate(va);
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

__global__ static void f_pointwiseMultiplyAndScale(
		cufftComplex* r, 
		const cufftComplex* ir1, const cufftComplex* ir2, 
		const cufftComplex* a1, const cufftComplex* a2, 
		size_t n, float scale1, float scale2)
{
	auto stride = gridDim * blockDim;
	auto offset = blockDim * blockIdx + threadIdx;

	for (auto s = offset.x; s < n; s += stride.x)
	{
		auto va1 = a1[s];
		auto va2 = a2[s];
		auto vir1 = ir1[s];
		auto vir2 = ir2[s];
		auto re1 = va1.x * vir1.x - va1.y * vir1.y;
		auto re2 = va2.x * vir2.x - va2.y * vir2.y;
		auto im1 = (va1.x + va1.y) * (vir1.x + vir1.y) - re1;
		auto im2 = (va2.x + va2.y) * (vir2.x + vir2.y) - re2;
		r[s] = make_float2(re1, im1) * scale1 + make_float2(re2,im2) * scale2;
	}
}


__global__ static void f_addDryInterleaved(
		cufftComplex* L, cufftComplex* R, 
		const cufftComplex* original, size_t n, 
		float scaleL1, float scaleR1, float scaleL2, float scaleR2)
{
	auto stride = gridDim * blockDim;
	auto offset = blockDim * blockIdx + threadIdx;

	for (auto s = offset.x; s < n; s += stride.x)
	{
		auto v = original[s];
		L[s] += v.x * scaleL1 + v.y * scaleL2;
		R[s] += v.x * scaleR1 + v.y * scaleR2;
	}
}

Convolution::Convolution(const std::string& name, size_t fftSize) : 
	JackClient(name),
	_fftSize(fftSize),
	capture{nullptr},
	playback{nullptr}
{
	int rc;
	for (auto i = 0; i<4; i++) 
	{
		rc = cudaStreamCreateWithFlags(&_streams[i], cudaStreamNonBlocking);
		assert(0 == rc);
	}

	cufftComplex** pcc[] = {
		&cin, &cin1, &cin2, &cinFFT, 
		&ir.left, &ir.right,
		&irFFT1.left, &irFFT1.right,
		&irFFT2.left, &irFFT2.right,
	};

	for (auto i = 0UL; i < sizeof(pcc) / sizeof(*pcc); i++)
	{
		rc = cudaMalloc(pcc[i], fftSize * sizeof(cufftComplex));
		assert(cudaSuccess == rc);
	}
		
	// TODO make this just float
	rc = cudaMalloc(&residual.left, (fftSize + CONV_MAX_PREDELAY) * sizeof(cufftComplex));
	assert(cudaSuccess == rc);

	rc = cudaMalloc(&residual.right, (fftSize + CONV_MAX_PREDELAY) * sizeof(cufftComplex));
	assert(cudaSuccess == rc);
	
	rc = cudaMalloc(&output.left, (fftSize + CONV_MAX_PREDELAY) * sizeof(cufftComplex));
	assert(cudaSuccess == rc);

	rc = cudaMalloc(&output.right, (fftSize + CONV_MAX_PREDELAY) * sizeof(cufftComplex));
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
}

void Convolution::onStart()
{
	activate();
	playback[0] = addOutput("playback_1");
	playback[1] = addOutput("playback_2");
	capture[0] = addInput("capture_1");
	capture[1] = addInput("capture_2");
}

// TODO, make thread safe
void Convolution::prepare(size_t idx, const WavFile& wav, size_t nframes)
{
	int rc;
	
	cudaStream_t stream = _streams[0];
	cufftSetStream(_plan, stream);

	auto buf = _irBuffers[idx];
	if (buf) cudaFree(buf);

	cufftComplex* tmp = nullptr;

#if 1
#define CONV_PREP_INPLACE 0
#define CONV_PREP_SRC tmp
	// Trying to solve a bug with certain fftSizes
	rc = cudaMalloc(&CONV_PREP_SRC, sizeof(cufftComplex) * (_fftSize));
	assert(cudaSuccess == rc);

	rc = cudaMemset(CONV_PREP_SRC, 0, sizeof(cufftComplex) * _fftSize);
	assert(cudaSuccess == rc);
#else
#define CONV_PREP_INPLACE 1
#define CONV_PREP_SRC buf
#endif
	// TODO pack real fft into half size buffer
	rc = cudaMalloc(&buf, sizeof(cufftComplex) * (_fftSize << 1));
	assert(cudaSuccess == rc);

#if !CONV_PREP_INPLACE
	cudaDeviceSynchronize();
#endif
	auto n = min(wav.numFrames, _fftSize - nframes);
	rc = cudaMemcpyAsync(CONV_PREP_SRC, wav.buffer, sizeof(cufftComplex) * n, cudaMemcpyDeviceToDevice, stream);
	assert(cudaSuccess == rc);
	
	rc = cufftExecC2C(_plan, CONV_PREP_SRC, CONV_PREP_SRC, CUFFT_FORWARD);
	assert(cudaSuccess == rc);
	
	f_unpackC22R <<< CONV_GRIDSIZE, CONV_BLOCKSIZE, 0, stream >>> (buf, buf+_fftSize, CONV_PREP_SRC,  _fftSize);

	cudaStreamSynchronize(stream);
#if !CONV_PREP_INPLACE
	cudaFree(CONV_PREP_SRC);
#endif
	_irBuffers[idx] = buf;
}

static void handleCC(Convolution::CC& cc, uint8_t m1, uint8_t m2, int v, size_t nb)
{
	if (cc.message == m1)
	{
		if (cc.select == m2) 
		{
			cc.value.select = v * nb / 0x80, cc.value.vsteps = cc.value.speed;
			Log::info("conv", "Selected IR %d", cc.value.select);
		}
		if (cc.predelay == m2) cc.value.predelay = v * CONV_MAX_PREDELAY / 0x80;
		if (cc.dry == m2) cc.value.dry = v / 128.0f;
		if (cc.wet == m2) cc.value.wet = v / 128.0f;
		if (cc.panDry == m2) cc.value.panDry = v / 64.0f - 1;
		if (cc.panWet == m2) cc.value.panWet = v / 64.0f - 1;
		if (cc.level == m2) cc.value.level = v / 128.0f;
		if (cc.speed == m2) 
		{
			cc.value.speed = (v * CONV_MAX_SPEED) / 0x80;
			if (cc.value.vsteps > cc.value.speed) cc.value.vsteps = cc.value.speed;
		}
	}
}

void Convolution::onMidiMessage(const RawMidi::Device* sender, const uint8_t* buffer, size_t len)
{
	if (len >= 3)
	{
		if (cc[0].device == sender) handleCC(cc[0], buffer[0], buffer[1], buffer[2], _irBuffers.size());
		if (cc[1].device == sender) handleCC(cc[1], buffer[0], buffer[1], buffer[2], _irBuffers.size());
	}
}

void Convolution::onProcess(size_t nframes)
{
	int rc;

	auto IN1 = capture[0] ? jack_port_get_buffer(capture[0], nframes) : nullptr;
	auto IN2 = capture[1] ? jack_port_get_buffer(capture[1], nframes) : nullptr;
	auto L = playback[0] ? jack_port_get_buffer(playback[0], nframes) : nullptr;
	auto R = playback[1] ? jack_port_get_buffer(playback[1], nframes) : nullptr;


	if (!IN1 || !IN2 || !L || !R) return;
	
	cudaEvent_t started, stopped;
	cudaEventCreate(&started);
	cudaEventCreate(&stopped);
	cudaEventRecord(started, _streams[0]);
	cufftSetStream(_plan, _streams[0]);

#if 0
	// copy input to device
	rc = cudaMemcpy2DAsync(
			cin,  sizeof(cufftComplex), 
			IN1,  sizeof(float), 
			sizeof(float), nframes,
			cudaMemcpyHostToDevice, _streams[1]);
	assert(cudaSuccess == rc);

	rc = cudaMemcpy2DAsync(
			((float*)cin)+1,  sizeof(cufftComplex), 
			IN2,  sizeof(float), 
			sizeof(float), nframes,
			cudaMemcpyHostToDevice, _streams[1]);
	assert(cudaSuccess == rc);
#else
	rc = cudaMemcpyAsync(cin1, IN1, sizeof(float) * nframes, cudaMemcpyHostToDevice, _streams[1]);
	assert(cudaSuccess == rc);
	rc = cudaMemcpyAsync(cin2, IN2, sizeof(float) * nframes, cudaMemcpyHostToDevice, _streams[1]);
	assert(cudaSuccess == rc);

	rc = cudaMemsetAsync(cin, 0, sizeof(cufftComplex) * _fftSize, _streams[1]);
	assert(cudaSuccess == rc);
	f_pack2R2C <<< CONV_GRIDSIZE, CONV_BLOCKSIZE, 0, _streams[1] >>> (cin, (float*)cin1, (float*)cin2, nframes);
#endif

#if CONV_INTERPOLATE
#define CONV_IRFFT1L irFFT1.left
#define CONV_IRFFT1R irFFT1.right
#define CONV_IRFFT2L irFFT2.left
#define CONV_IRFFT2R irFFT2.right

	// Nicer but one of the most time consuming parts
	// interpolate to IR FFT
	f_interpolate <<< CONV_GRIDSIZE, CONV_BLOCKSIZE, 0, _streams[2] >>> (
			irFFT1.left, irFFT1.left, _irBuffers[cc[0].value.select], 
			_fftSize, cc[0].value.vsteps, cc[0].value.wet);
	f_interpolate <<< CONV_GRIDSIZE, CONV_BLOCKSIZE, 0, _streams[3] >>> (
			irFFT1.right, irFFT1.right, _irBuffers[cc[0].value.select]+_fftSize, 
			_fftSize, cc[0].value.vsteps, cc[0].value.wet);
	if (cc[0].value.vsteps > 0) cc[0].value.vsteps--;

	f_interpolate <<< CONV_GRIDSIZE, CONV_BLOCKSIZE, 0, _streams[2] >>> (
			irFFT2.left, irFFT2.left, _irBuffers[cc[1].value.select], 
			_fftSize, cc[1].value.vsteps, cc[1].value.wet);
	f_interpolate <<< CONV_GRIDSIZE, CONV_BLOCKSIZE, 0, _streams[3] >>> (
			irFFT2.right, irFFT2.right, _irBuffers[cc[1].value.select]+_fftSize, 
			_fftSize, cc[1].value.vsteps, cc[1].value.wet);
	if (cc[1].value.vsteps > 0) cc[1].value.vsteps--;
#else
#warning Beware. Changing IR during live play may cause some noise.
#define CONV_IRFFT1L (_irBuffers[cc[0].value.select])
#define CONV_IRFFT1R (_irBuffers[cc[0].value.select]+_fftSize)
#define CONV_IRFFT2L (_irBuffers[cc[1].value.select])
#define CONV_IRFFT2R (_irBuffers[cc[1].value.select]+_fftSize)

#endif

	rc = cudaStreamSynchronize(_streams[1]);
	assert(cudaSuccess == rc);

	// get FFT of input
	rc = cufftExecC2C(_plan, cin, cinFFT, CUFFT_FORWARD);
	assert(cudaSuccess == rc);
	
	f_unpackC22R <<< CONV_GRIDSIZE, CONV_BLOCKSIZE, 0, _streams[0] >>> (
			cin1, cin2, cinFFT, _fftSize);

#if 0 // TODO this interferes with the interpolate above
#if CONV_LOWPASS
	f_lowpass <<< CONV_GRIDSIZE, CONV_BLOCKSIZE, 0, _streams[0] >>> (
			CONV_IRFFT1L, CONV_IRFFT1L, _fftSize);
	f_lowpass <<< CONV_GRIDSIZE, CONV_BLOCKSIZE, 0, _streams[0] >>> (
			CONV_IRFFT1R, CONV_IRFFT1R, _fftSize);
	f_lowpass <<< CONV_GRIDSIZE, CONV_BLOCKSIZE, 0, _streams[0] >>> (
			CONV_IRFFT2L, CONV_IRFFT2L, _fftSize);
	f_lowpass <<< CONV_GRIDSIZE, CONV_BLOCKSIZE, 0, _streams[0] >>> (
			CONV_IRFFT2R, CONV_IRFFT2R, _fftSize);
#endif
#endif
	// multiply ir with input
	float panL1 = cc[0].value.panWet >= 0 ? 1 - cc[0].value.panWet : 1;
	float panR1 = cc[0].value.panWet <= 0 ? 1 + cc[0].value.panWet : 1;
	float panL2 = cc[1].value.panWet >= 0 ? 1 - cc[1].value.panWet : 1;
	float panR2 = cc[1].value.panWet <= 0 ? 1 + cc[1].value.panWet : 1;

	cudaStreamSynchronize(_streams[2]);
	f_pointwiseMultiplyAndScale <<< CONV_GRIDSIZE, CONV_BLOCKSIZE, 0, _streams[0] >>> (
			output.left, CONV_IRFFT1L, CONV_IRFFT2L, cin1, cin2, _fftSize, 
			1.0f/_fftSize * panL1 * cc[0].value.level, 
			1.0f/_fftSize * panL2 * cc[1].value.level);
	
	cudaStreamSynchronize(_streams[3]);
	f_pointwiseMultiplyAndScale <<< CONV_GRIDSIZE, CONV_BLOCKSIZE, 0, _streams[0] >>> (
		 	output.right, CONV_IRFFT1R, CONV_IRFFT2R, cin1, cin2, _fftSize, 
			1.0f/_fftSize * panR1 * cc[0].value.level,
			1.0f/_fftSize * panR2 * cc[1].value.level);

	auto tmp = ir;
	// take the inverse FFT of the output
	rc = cufftExecC2C(_plan, output.left, tmp.left, CUFFT_INVERSE);
	assert(cudaSuccess == rc);
	rc = cufftExecC2C(_plan, output.right, tmp.right, CUFFT_INVERSE);
	assert(cudaSuccess == rc);
		
	// Add the residual
	f_pointwiseAdd <<< CONV_GRIDSIZE, CONV_BLOCKSIZE, 0, _streams[0] >>> (
			output.left, residual.left, tmp.left, _fftSize, cc[0].value.predelay);
	
	f_pointwiseAdd <<< CONV_GRIDSIZE, CONV_BLOCKSIZE, 0, _streams[0] >>> (
			output.right, residual.right, tmp.right, _fftSize, cc[0].value.predelay);

	// Add dry signal, cin still interleaved
	panL1 = cc[0].value.panDry >= 0 ? 1 - cc[0].value.panDry : 1;
	panR1 = cc[0].value.panDry <= 0 ? 1 + cc[0].value.panDry : 1;
	panL2 = cc[1].value.panDry >= 0 ? 1 - cc[1].value.panDry : 1;
	panR2 = cc[1].value.panDry <= 0 ? 1 + cc[1].value.panDry : 1;
	f_addDryInterleaved <<< 1, CONV_BLOCKSIZE, 0, _streams[0] >>> (
			output.left, output.right, cin, nframes, 
			cc[0].value.dry * panL1 * cc[0].value.level, 
			cc[0].value.dry * panR1 * cc[0].value.level,
			cc[1].value.dry * panL2 * cc[1].value.level, 
			cc[1].value.dry * panR2 * cc[1].value.level);


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
			(_fftSize + CONV_MAX_PREDELAY - nframes) * sizeof(cufftComplex), 
			cudaMemcpyDeviceToDevice, _streams[0]);
	assert(cudaSuccess == rc);
	rc = cudaMemcpyAsync(
			residual.right, 
			output.right + nframes, 
			(_fftSize + CONV_MAX_PREDELAY - nframes) * sizeof(cufftComplex), 
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

