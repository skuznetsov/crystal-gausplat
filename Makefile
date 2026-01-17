# Makefile for 3D Scanner (Gaussian Splatting)

CRYSTAL = crystal
CRYSTAL_FLAGS = --release -Duse_metal

# Source files
SOURCES = $(shell find src -name "*.cr")
METAL_SOURCES = $(shell find src/metal/kernels -name "*.metal")
OBJC_SOURCES = src/metal/bridge.mm
FFMPEG_SOURCES = src/video/ffmpeg_bridge.c

# Output
OUTPUT = gsplat
METAL_LIB = build/kernels.metallib

# Directories
BUILD_DIR = build

# FFmpeg paths (Homebrew)
FFMPEG_CFLAGS = $(shell pkg-config --cflags libavformat libavcodec libavutil libswscale 2>/dev/null || echo "-I/opt/homebrew/include")
FFMPEG_LIBS = $(shell pkg-config --libs libavformat libavcodec libavutil libswscale 2>/dev/null || echo "-L/opt/homebrew/lib -lavformat -lavcodec -lavutil -lswscale")

.PHONY: all clean release debug test run spec check-ffmpeg

all: release

# Check FFmpeg installation
check-ffmpeg:
	@which ffmpeg > /dev/null || (echo "FFmpeg not found. Install with: brew install ffmpeg" && exit 1)
	@pkg-config --exists libavformat || echo "Warning: pkg-config can't find FFmpeg, using default paths"

# Build release version
release: $(OUTPUT)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Compile Metal shaders to library (optional, can JIT compile from source)
$(METAL_LIB): $(METAL_SOURCES)
	xcrun -sdk macosx metal -c $(METAL_SOURCES) -o $(BUILD_DIR)/kernels.air
	xcrun -sdk macosx metallib $(BUILD_DIR)/kernels.air -o $(METAL_LIB)

# Build Objective-C++ bridge
$(BUILD_DIR)/bridge.o: $(OBJC_SOURCES) | $(BUILD_DIR)
	clang++ -c $(OBJC_SOURCES) -o $(BUILD_DIR)/bridge.o \
		-std=c++17 -fobjc-arc -fPIC

# Build FFmpeg bridge
$(BUILD_DIR)/ffmpeg_bridge.o: $(FFMPEG_SOURCES) | $(BUILD_DIR)
	clang -c $(FFMPEG_SOURCES) -o $(BUILD_DIR)/ffmpeg_bridge.o \
		$(FFMPEG_CFLAGS) -fPIC

# Build main executable
$(OUTPUT): $(SOURCES) $(BUILD_DIR)/bridge.o $(BUILD_DIR)/ffmpeg_bridge.o
	$(CRYSTAL) build src/main.cr -o $(OUTPUT) $(CRYSTAL_FLAGS) \
		--link-flags="$(shell pwd)/$(BUILD_DIR)/bridge.o $(shell pwd)/$(BUILD_DIR)/ffmpeg_bridge.o -framework Metal -framework Foundation -lc++ $(FFMPEG_LIBS)"

# Debug build
debug: $(BUILD_DIR) $(BUILD_DIR)/bridge.o $(BUILD_DIR)/ffmpeg_bridge.o
	$(CRYSTAL) build src/main.cr -o $(OUTPUT)_debug \
		--link-flags="$(shell pwd)/$(BUILD_DIR)/bridge.o $(shell pwd)/$(BUILD_DIR)/ffmpeg_bridge.o -framework Metal -framework Foundation -lc++ $(FFMPEG_LIBS)"

# Run tests (old internal test)
test: release
	./$(OUTPUT) test

# Run Crystal specs
spec: $(BUILD_DIR)/bridge.o $(BUILD_DIR)/ffmpeg_bridge.o
	$(CRYSTAL) spec \
		--link-flags="$(shell pwd)/$(BUILD_DIR)/bridge.o $(shell pwd)/$(BUILD_DIR)/ffmpeg_bridge.o -framework Metal -framework Foundation -lc++ $(FFMPEG_LIBS)"

# Run with arguments
run: release
	./$(OUTPUT) $(ARGS)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
	rm -f $(OUTPUT) $(OUTPUT)_debug

# Format code
format:
	crystal tool format src

# Check code style
lint:
	bin/ameba src

# Install dependencies
deps:
	shards install

# Help
help:
	@echo "Available targets:"
	@echo "  release - Build release version (default)"
	@echo "  debug   - Build debug version"
	@echo "  test    - Run tests"
	@echo "  run     - Run with ARGS='...'"
	@echo "  clean   - Clean build artifacts"
	@echo "  format  - Format source code"
	@echo "  lint    - Check code style"
	@echo "  deps    - Install dependencies"
