NVCC         := nvcc 
NVCCFLAGS    := -O3 -std=c++11 -Xptxas=-v -arch=sm_61
NVCCLDFLAGS  := -L/home/pp19/share/lib -Xlinker=-rpath,/home/pp19/share/lib -llab5

SRC := sobel.cu
EXE := $(SRC:cu=cuda)

TESTCASE     := candy
TESTCASE_DIR := ./cases
INPUT_PNG    := $(TESTCASE_DIR)/$(TESTCASE).png
OUTPUT_PNG   := out
EXPECT_PNG   := $(TESTCASE_DIR)/$(TESTCASE).out.png

TARGETS := $(EXE)

.PHONY: all
all: $(TARGETS)

$(EXE): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(NVCCLDFLAGS) -o $@ $<

.PHONY: run
run: $(EXE)
	./$< $(INPUT_PNG) $(OUTPUT_PNG)
	lab5-diff $(OUTPUT_PNG) $(EXPECT_PNG)

.PHONY: clean
clean:
	rm -f $(TARGETS)
