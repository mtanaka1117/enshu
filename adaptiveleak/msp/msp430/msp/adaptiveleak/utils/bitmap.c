#include "bitmap.h"


void set_bit(uint16_t index, struct BitMap *bitmap) {    
    uint16_t byteIndex = index >> 3;
    uint8_t bitOffset = index & 0x7;
    bitmap->bytes[byteIndex] |= (1 << bitOffset);
}


void unset_bit(uint16_t index, struct BitMap *bitmap) {
    uint16_t byteIndex = index >> 3;
    uint8_t bitOffset = index & 0x7;
    bitmap->bytes[byteIndex] &= ~(1 << bitOffset);
}


void clear_bitmap(struct BitMap *bitmap) {
    uint16_t idx = bitmap->numBytes;
    for (; idx > 0; idx--) {
        bitmap->bytes[idx - 1] = 0;
    }
}
