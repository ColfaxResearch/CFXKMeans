# common flags
ARFLAGS=-fPIC
COMMONFLAGS=-std=c++11 -O3 

########### Intel compiler ###################
CC=icc
CXX = icpc

### Enable vectorization support
## Intel Xeon Phi processor        -> -xMIC_AVX512
## Intel Xeon processor (HSW, BDW) -> -xCORE_AVX2
## Intel Xeon processor (SNB, IVB) -> -xAVX
VECFLAG=-xMIC_AVX512

CXXFLAGS=$(COMMONFLAGS) $(VECFLAG) -qopenmp

########### GCC (> 4.9.1) ####################
#CC=gcc
#CXX = g++
#
#### Enable vectorization support
### Intel Xeon Phi processor        -> -mavx512f -mavx512cd -mavx512pf 
### Intel Xeon processor (HSW, BDW) -> -mavx2
### Intel Xeon processor (SNB, IVB) -> -mavx
#VECFLAG=-mavx512f -mavx512cd -mavx512pf
#
#CXXFLAGS=$(COMMONFLAGS) $(VECFLAG) -fopenmp -ffast-math
#
#############################################

# Do not change these two
LIBDIR=lib
SRCDIR=src

ARCHIVE=$(LIBDIR)/libcfxkmeans.a
SETUPFILE=setup.py
OBJECTS=$(LIBDIR)/cfxkmeans.o\
        $(LIBDIR)/cfxkmeans-hamerly.o

default: $(ARCHIVE)

all: $(ARCHIVE) python-build

$(ARCHIVE): mkdirlib $(OBJECTS)
	ar rvs  $(ARCHIVE) $(OBJECTS)

lib/cfxkmeans.o: src/cfxkmeans.cc
	$(CXX) -c $(ARFLAGS) $(CXXFLAGS) -o "$@" "$<"

lib/%.o: src/%.cc
	$(CXX) -c $(ARFLAGS) $(CXXFLAGS) -DDTYPE=float  -o "$@"-sp "$<"
	$(CXX) -c $(ARFLAGS) $(CXXFLAGS) -DDTYPE=double -o "$@"-dp "$<"
	ld -r "$@-dp" "$@-sp" -o $@
	rm "$@-dp" "$@-sp"
	
mkdirlib:
	mkdir -p $(LIBDIR)

python-build: $(ARCHIVE)
	CC=$(CC) CXX=$(CXX) python $(SETUPFILE) build --force
python-install: $(ARCHIVE)
	CC=$(CC) CXX=$(CXX) python $(SETUPFILE) install

clean:
	rm -f $(ARCHIVE) $(OBJECTS)
	rmdir $(LIBDIR)
