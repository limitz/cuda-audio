TARGET = program

PROC = $(shell uname -m)
ARCH = $(PROC)-linux

ifeq ($(PROC), x86_64)
CFLAGS := -std=c++17 -O3 -fPIC -Wall
IFLAGS := -I/usr/local/cuda/include
LFLAGS := -rpath='$$ORIGIN'
LFLAGS += -L/usr/local/cuda/lib64 -lcudart -lcufft -ljack
SMS := 53 61 86
HIGHEST_SM = $(lastword $(sort $(SMS)))
NVCCFLAGS := -m64 -rdc=true -std=c++17 
NVCCFLAGS += $(foreach flag,$(CFLAGS),$(addprefix -Xcompiler ,$(flag))) $(IFLAGS)
NVLDFLAGS := -m64 
NVLDFLAGS += $(foreach flag,$(LFLAGS),$(addprefix -Xlinker ,$(flag)))
GENCODE := 
$(foreach sm,$(SMS), $(eval GENCODE += -gencode arch=compute_$(sm),code=sm_$(sm)))
GENCODE += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
else
CFLAGS := -O3 -fPIC -Wall -DCONV_FFTSIZE=98304 -DCONV_GRIDSIZE=64 -DCONV_BLOCKSIZE=128
IFLAGS := -I/usr/local/cuda/include
LFLAGS := -rpath='$$ORIGIN'
LFLAGS += -L/usr/local/cuda/lib64 -lcudart -lcufft -ljack
SMS := 53 72 
HIGHEST_SM = $(lastword $(sort $(SMS)))
NVCCFLAGS := -m64 -rdc=true 
NVCCFLAGS += $(foreach flag,$(CFLAGS),$(addprefix -Xcompiler ,$(flag))) $(IFLAGS)
NVLDFLAGS := -m64 
NVLDFLAGS += $(foreach flag,$(LFLAGS),$(addprefix -Xlinker ,$(flag)))
GENCODE := 
$(foreach sm,$(SMS), $(eval GENCODE += -gencode arch=compute_$(sm),code=sm_$(sm)))
GENCODE += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif

SRC_DIR := src
BIN_DIR := bin
OBJ_DIR := obj

NVCC := nvcc $(NVCCFLAGS)
NVLD := nvcc $(NVLDFLAGS)

INCLUDES := $(wildcard $(SRC_DIR)/*.h)
SOURCES := $(wildcard $(SRC_DIR)/*.cu)
OBJECTS := $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(SOURCES))

$(BIN_DIR)/$(TARGET): $(OBJECTS)
	mkdir -p $(BIN_DIR)
	$(NVLD) -o $@ $(OBJECTS) $(GENCODE)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(INCLUDES) 
	mkdir -p $(OBJ_DIR)
	$(NVCC) -c $< -o $@

.PHONY: clean
clean:
	rm -rf $(OBJ_DIR)
	rm -rf $(BIN_DIR)


