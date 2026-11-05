#pragma once

// Select this specialization from the shared head64 implementation.
#define FLASH_MLA_SMALL_HEADS 0
#include "../head64_shared/config.h"
#undef FLASH_MLA_SMALL_HEADS
