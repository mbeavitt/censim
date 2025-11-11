CC = gcc
CFLAGS = -O3 -march=native -Wall
TARGET = main
SRC = kmer_variance.c
SHARED_LIB = libkmer.so
SHARED_LIB_MACOS = libkmer.dylib

# Detect OS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    SHARED_TARGET = $(SHARED_LIB_MACOS)
    SHARED_FLAGS = -dynamiclib
else
    SHARED_TARGET = $(SHARED_LIB)
    SHARED_FLAGS = -shared
endif

.PHONY: all clean shared

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

shared: $(SHARED_TARGET)

$(SHARED_LIB): $(SRC)
	$(CC) $(CFLAGS) -fPIC $(SHARED_FLAGS) -o $(SHARED_LIB) $(SRC)

$(SHARED_LIB_MACOS): $(SRC)
	$(CC) $(CFLAGS) -fPIC $(SHARED_FLAGS) -o $(SHARED_LIB_MACOS) $(SRC)

clean:
	rm -f $(TARGET) $(SHARED_LIB) $(SHARED_LIB_MACOS)
