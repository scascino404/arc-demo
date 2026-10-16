# arc-demo native build: physics/simulation library, renderer, tools, tests.

CC      ?= gcc
CSTD    := -std=c89 -pedantic -Wall -Wextra
# -march=native and FMA contraction (-ffp-contract=fast) are worth ~13% in the
# solver; build with e.g. `make OPT="-O3 -g"` for binaries that run elsewhere.
OPT     ?= -O3 -march=native -ffp-contract=fast -g
OMP     := -fopenmp
CFLAGS  := $(CSTD) $(OPT) -fno-math-errno -fno-trapping-math -fPIC
LDLIBS  := -lm

BUILD   := build

SDL_CFLAGS := $(shell pkg-config --cflags sdl2)
SDL_LIBS   := $(shell pkg-config --libs sdl2) -lGL

SIM_SRC := csrc/arc_phys.c csrc/arc_robot.c csrc/arc_terrain.c csrc/arc_sim.c
SIM_OBJ := $(SIM_SRC:csrc/%.c=$(BUILD)/%.o)

.PHONY: all test bench clean

all: arc/libarcsim.so arc/libarcrender.so bin/arcview

$(BUILD)/%.o: csrc/%.c csrc/*.h | $(BUILD)
	$(CC) $(CFLAGS) $(OMP) -c $< -o $@

$(BUILD):
	mkdir -p $@

arc/libarcsim.so: $(SIM_OBJ)
	$(CC) -shared $(OMP) $^ -o $@ $(LDLIBS)

$(BUILD)/arc_render.o: csrc/arc_render.c csrc/*.h | $(BUILD)
	$(CC) $(CFLAGS) $(SDL_CFLAGS) -c $< -o $@

# the renderer finds libarcsim.so next to itself
arc/libarcrender.so: $(BUILD)/arc_render.o arc/libarcsim.so
	$(CC) -shared $< -o $@ -Larc -larcsim $(SDL_LIBS) $(LDLIBS) -Wl,-rpath,'$$ORIGIN'

bin/arcview: csrc/tools/arcview.c arc/libarcrender.so arc/libarcsim.so | bin
	$(CC) $(CFLAGS) $(SDL_CFLAGS) $< -o $@ -Larc -larcrender -larcsim $(SDL_LIBS) $(LDLIBS) \
		-Wl,-rpath,'$$ORIGIN/../arc'

bin:
	mkdir -p $@

# test programs and the benchmark link the simulation objects statically
$(BUILD)/test_%: csrc/tests/test_%.c csrc/tests/check.h $(SIM_OBJ)
	$(CC) $(CFLAGS) $(OMP) $< $(SIM_OBJ) -o $@ $(LDLIBS)

$(BUILD)/bench: csrc/tools/bench.c $(SIM_OBJ)
	$(CC) $(CFLAGS) $(OMP) $^ -o $@ $(LDLIBS)

test: $(BUILD)/test_phys $(BUILD)/test_robot
	$(BUILD)/test_phys
	$(BUILD)/test_robot

bench: $(BUILD)/bench
	$(BUILD)/bench

clean:
	rm -rf $(BUILD) bin arc/*.so
