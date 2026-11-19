#pragma once

// Select this specialization from the shared head64 implementation.
#define FLASH_MLA_SMALL_HEADS 1
// The sm_107 copy (namespace small_heads_sm107) first: the default copy's entry point dispatches to it
#define FLASH_MLA_TARGET 107
#include "../head64_shared/config.h"
#undef FLASH_MLA_TARGET
#include "../head64_shared/config.h"
#undef FLASH_MLA_SMALL_HEADS
