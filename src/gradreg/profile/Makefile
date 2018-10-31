OS := $(shell uname)
SRCS := $(wildcard *.py)
OBJS := $(SRCS:.py=.mem.txt) $(SRCS:.py=.time.txt)
CUDASRCS := $(wildcard *gpu.py)
CUDAOBJS := $(CUDASRCS:.py=.cuda.txt)

nvprof := /usr/local/cuda/bin/nvprof

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
	$(nvprof) --profile-child-processes python $< 2> $@
	$(call postproc_profile_result,$@)

%gpu.precuda.py: %gpu.py
	sed '/@profile/d' $< > $@

clean:
	rm -f -- $(wildcard *.py.prof)
	rm -f -- $(wildcard *gpu.precuda.py)
