ARCH ?= sm_75
NVCC ?= nvcc
CC ?= gcc
CFLAGS ?= -O2
NVCCFLAGS ?= -O2

CUDA_SRC := $(wildcard src/*.cu)
C_SRC := $(wildcard src/*.c)

CUDA_BIN := $(patsubst src/%.cu,bin/%_cuda,$(CUDA_SRC))
C_BIN := $(patsubst src/%.c,bin/%_cpu,$(C_SRC))

.PHONY: all cpu cuda run run-cpu run-cuda clean

all: cpu cuda

cpu: $(C_BIN)

cuda: $(CUDA_BIN)

bin:
	mkdir -p bin

bin/%_cpu: src/%.c | bin
	$(CC) $(CFLAGS) $< -o $@

bin/%_cuda: src/%.cu | bin
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@

run-cpu: cpu
	@for f in $(C_BIN); do \
		echo ""; \
		echo "=============================="; \
		echo "Ejecutando $$f"; \
		echo "=============================="; \
		./$$f; \
	done

run-cuda: cuda
	@for f in $(CUDA_BIN); do \
		echo ""; \
		echo "=============================="; \
		echo "Ejecutando $$f"; \
		echo "=============================="; \
		./$$f; \
	done

run: all
	@for f in $(C_BIN) $(CUDA_BIN); do \
		echo ""; \
		echo "=============================="; \
		echo "Ejecutando $$f"; \
		echo "=============================="; \
		./$$f; \
	done

clean:
	rm -rf bin
