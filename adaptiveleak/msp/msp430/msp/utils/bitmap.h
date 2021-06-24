#include <stdint.h>
#include "constants.h"


#ifndef BITMAP_H_
#define BITMAP_H_

struct BitMap {
    uint8_t *bytes;
    uint16_t numBytes;
};

void set_bit(uint16_t index, struct BitMap *bitmap);
void unset_bit(uint16_t index, struct BitMap *bitmap);
void clear_bitmap(struct BitMap *bitmap);

#endif
