#pragma once
#include "phase1.h"
#include "config.h"

// Select this specialization from the shared head64 implementation.
#define FLASH_MLA_SMALL_HEADS 0
// The sm_107 copy (namespace head64_sm107) first: the default copy's entry point dispatches to it
#define FLASH_MLA_TARGET 107
#include "../head64_shared/phase1.cuh"
#undef FLASH_MLA_TARGET
#include "../head64_shared/phase1.cuh"
#undef FLASH_MLA_SMALL_HEADS
