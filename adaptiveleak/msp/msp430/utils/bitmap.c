#include "bitmap.h"


void set_bit(uint16_t index, struct BitMap *bitmap) {    
    uint16_t byteIndex = index / BITS_PER_BYTE;
    uint16_t bitOffset = index - (byteIndex * BITS_PER_BYTE);
    bitmap->bytes[byteIndex] |= (1 << bitOffset);
}


void clear_bitmap(struct BitMap *bitmap) {
    uint16_t idx = bitmap->numBytes;
    for (; idx > 0; idx--) {
        bitmap->bytes[idx - 1] = 0;
    }
}
