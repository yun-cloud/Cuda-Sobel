NVCC         = nvcc 
NVCCFLAGS    = -O3 -std=c++11 -Xptxas=-v -arch=sm_61
NVCCLDFLAGS  = -L/home/pp19/share/lib -Xlinker=-rpath,/home/pp19/share/lib -llab5

TARGETS = sobel.cuda

.PHONY: all
all: $(TARGETS)

sobel.cuda: sobel.cu
	$(NVCC) $(NVCCFLAGS) $(NVCCLDFLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -f $(TARGETS)
