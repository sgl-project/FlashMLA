#pragma once
#include "phase1.h"
#include "config.h"

// Select this specialization from the shared head64 implementation.
#define FLASH_MLA_SMALL_HEADS 1
#include "../head64_shared/phase1.cuh"
#undef FLASH_MLA_SMALL_HEADS
