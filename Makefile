NVCC         := nvcc 
NVCCFLAGS    := -O3 -std=c++11 -Xptxas=-v -arch=sm_61
NVCCLDFLAGS  := -L/home/pp19/share/lib -Xlinker=-rpath,/home/pp19/share/lib -llab5

SRC := sobel.cu
EXE := $(SRC:cu=cuda)

# TESTCASE ?= large-candy
TESTCASE ?= candy
TESTCASE_DIR := ./cases
INPUT_IMG  := $(TESTCASE_DIR)/$(TESTCASE).png
EXPECT_IMG := $(TESTCASE_DIR)/$(TESTCASE).out.png
OUTPUT_IMG := out.png

TARGETS := $(EXE)

.PHONY: all
all: $(TARGETS)

$(EXE): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(NVCCLDFLAGS) -o $@ $<

.PHONY: run
run: $(EXE)
	$(RM) $(OUTPUT_IMG)
	srun -n1 -p pp --gres=gpu:1 time ./$< $(INPUT_IMG) $(OUTPUT_IMG)
	@#srun -n1 -p pp --gres=gpu:1 nvprof ./$< $(INPUT_IMG) $(OUTPUT_IMG) 2>&1 | rg 'GPU activities'
	lab5-diff $(EXPECT_IMG) $(OUTPUT_IMG)

.PHONY: run2
run2: $(EXE)
	$(RM) $(OUTPUT_IMG)
	CUDA_VISIBLE_DEVICES=1 ./$< $(INPUT_IMG) $(OUTPUT_IMG)
	@#CUDA_VISIBLE_DEVICES=1 nvprof ./$< $(INPUT_IMG) $(OUTPUT_IMG) 2>&1 | rg 'GPU activities'
	@#CUDA_VISIBLE_DEVICES=1 nvprof --metrics shared_load_transactions_per_request,shared_store_transactions_per_request  ./$< $(INPUT_IMG) $(OUTPUT_IMG) 2>&1
	@#CUDA_VISIBLE_DEVICES=1 cuda-memcheck ./$< $(INPUT_IMG) $(OUTPUT_IMG)
	lab5-diff $(EXPECT_IMG) $(OUTPUT_IMG)

.PHONY: clean
clean:
	rm -f $(TARGETS)
