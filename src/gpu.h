#pragma once

#include <cuda_runtime.h>
#include <pthread.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <cstdint>
#include <cassert>
#include "operators.h"
#include "log.h"

#define warpSize 32
#define warpMask 0x1F
#define warpIdx(thread) ((thread) >> 5)

void selectGpu();
