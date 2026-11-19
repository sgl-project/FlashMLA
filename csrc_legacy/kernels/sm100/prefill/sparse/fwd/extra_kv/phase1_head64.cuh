#pragma once
#include "phase1.h"

// Select the extra-KV-cache variant of the shared head64 implementation, with the head64 head layout.
#define FLASH_MLA_SMALL_HEADS 0
#define FLASH_MLA_EXTRA_KV 1
// The sm_107 copy (namespace extra_kv::head64_sm107) first: the default copy's entry point dispatches to it
#define FLASH_MLA_TARGET 107
#include "../head64_shared/config.h"
#include "../head64_shared/phase1.cuh"
#undef FLASH_MLA_TARGET
#include "../head64_shared/config.h"
#include "../head64_shared/phase1.cuh"
#undef FLASH_MLA_EXTRA_KV
#undef FLASH_MLA_SMALL_HEADS
