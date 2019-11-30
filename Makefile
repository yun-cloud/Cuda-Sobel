NVCC         := nvcc 
NVCCFLAGS    := -O3 -std=c++11 -Xptxas=-v -arch=sm_61
NVCCLDFLAGS  := -L/home/pp19/share/lib -Xlinker=-rpath,/home/pp19/share/lib -llab5

SRC := sobel.cu
EXE := $(SRC:cu=cuda)

TESTCASE ?= candy
TESTCASE_DIR := ./cases
INPUT_IMG  := $(TESTCASE_DIR)/$(TESTCASE).png
EXPECT_IMG := $(TESTCASE_DIR)/$(TESTCASE).out.png
OUTPUT_IMG := out

TARGETS := $(EXE)

.PHONY: all
all: $(TARGETS)

$(EXE): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(NVCCLDFLAGS) -o $@ $<

.PHONY: run
run: $(EXE)
	$(RM) $(OUTPUT_IMG)
	srun -n1 -p pp --gres=gpu:1 time ./$< $(INPUT_IMG) $(OUTPUT_IMG)
	lab5-diff $(EXPECT_IMG) $(OUTPUT_IMG)

.PHONY: clean
clean:
	rm -f $(TARGETS)
