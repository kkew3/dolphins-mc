OS := $(shell uname)
SRCS := $(wildcard *.py)
OBJS := $(SRCS:.py=.mem.txt) $(SRCS:.py=.time.txt)
CUDASRCS := $(wildcard *gpu.py)
CUDAOBJS := $(CUDASRCS:.py=.cuda.txt)

nvprof := /usr/local/cuda/bin/nvprof
#NVPROFFLAGS := --profile-child-processes --track-memory-allocations on
# with "--track-memory-allocations on" it somehow fails
NVPROFFLAGS := --profile-child-processes

.PHONY: all cuda clean

all: $(OBJS)

cuda: $(CUDAOBJS)

ifeq ($(OS),Darwin)
define postproc_profile_result
	sed -i '' 's/  *$$//' $(1)
	sed -i '' '/^$$/d' $(1)
endef
else
define postproc_profile_result
	sed -i 's/  *$$//' $(1)
	sed -i '/^$$/d' $(1)
endef
endif


%.mem.txt: %.py
	grep -q '@profile' $<
	! grep -q '\<pdb\>' $<
	python -m memory_profiler $< > $@
	$(call postproc_profile_result,$@)

%.time.txt: %.py
	grep -q '@profile' $<
	! grep -q '\<pdb\>' $<
	kernprof -l $<
	python -m line_profiler $<.lprof > $@
	$(call postproc_profile_result,$@)

%gpu.cuda.txt: %gpu.precuda.py
	$(nvprof) $(NVPROFFLAGS) python $< 2> $@
	$(call postproc_profile_result,$@)

%gpu.precuda.py: %gpu.py
	sed '/@profile/d' $< > $@
	sed -i '1s/^/import ctypes; _cudart = ctypes.CDLL("libcudart.so")\n/' $@
	sed -i "$$(grep -nA1 'if __name__ == \(['"'"'"]\)__main__\1' p1.py | cut -d':' -f1 | cut -d'-' -f1 | tail -1)"'s/^/    _cudart\.cudaProfilerStart()\n/' $@
	echo '    _cudart.cudaProfilerStop()' >> $@

clean:
	rm -f -- $(SRCS:.py=*.py.lprof)
	rm -f -- $(CUDASRCS:.py=.precuda.py)
