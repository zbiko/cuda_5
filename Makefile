# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -g -G -Xcompiler "-Wall -Wno-unused-function"

# Target executable name
TARGET = cuda_app

# CUDA source files
CUDA_SRC = main.cu

# Object files
OBJS = $(CUDA_SRC:.cu=.o)

# Libraries
LIBS = -lcuda

# Build rule for target
$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LIBS)

# Rule for object files
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Clean rule
.PHONY: clean
clean:
	rm -f $(OBJS) $(TARGET)
