# arc-demo native build: physics/simulation library, renderer, tools, tests.

CC      ?= gcc
CSTD    := -std=c89 -pedantic -Wall -Wextra
OPT     ?= -O2 -g
OMP     := -fopenmp
CFLAGS  := $(CSTD) $(OPT) -fPIC
LDLIBS  := -lm

BUILD   := build

SIM_SRC := csrc/arc_phys.c csrc/arc_robot.c csrc/arc_terrain.c csrc/arc_sim.c
SIM_OBJ := $(SIM_SRC:csrc/%.c=$(BUILD)/%.o)

.PHONY: all test bench clean

all: arc/libarcsim.so

$(BUILD)/%.o: csrc/%.c csrc/*.h | $(BUILD)
	$(CC) $(CFLAGS) $(OMP) -c $< -o $@

$(BUILD):
	mkdir -p $@

arc/libarcsim.so: $(SIM_OBJ)
	$(CC) -shared $(OMP) $^ -o $@ $(LDLIBS)

$(BUILD)/test_phys: csrc/tests/test_phys.c $(SIM_OBJ)
	$(CC) $(CFLAGS) $(OMP) $^ -o $@ $(LDLIBS)

$(BUILD)/test_robot: csrc/tests/test_robot.c $(SIM_OBJ)
	$(CC) $(CFLAGS) $(OMP) $^ -o $@ $(LDLIBS)

test: $(BUILD)/test_phys $(BUILD)/test_robot
	$(BUILD)/test_phys
	$(BUILD)/test_robot

$(BUILD)/bench: csrc/tools/bench.c $(SIM_OBJ)
	$(CC) $(CFLAGS) $(OMP) $^ -o $@ $(LDLIBS)

bench: $(BUILD)/bench
	$(BUILD)/bench

clean:
	rm -rf $(BUILD) bin arc/*.so
