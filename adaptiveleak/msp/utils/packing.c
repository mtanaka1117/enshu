

uint16_t *pack(uint8_t *output, struct FixedPoint *values, uint8_t bitWidth, uint16_t numValues) {
    /**
     * Packs the given features into the output byte array using the given bit width per feature.
     */
    if (numValues == 0) {
        return 0;
    }

    // Get the number of bytes for each value
    uint8_t bytesPerVal = bitWidth / BITS_PER_BYTE;
    uint16_t remainderBits = bitWidth - (BITS_PER_BYTE * bytesPerVal);

    if (reaminderBits > 0) {
        bytesPerVal++;
    }

    uint16_t current = 0;
    uint16_t currentByte = 0;
    uint16_t value = 0;
    uint16_t consumedBits = 0;

    uint16_t byteIdx, byteNum;
    uint16_t valIdx;

    for (valIdx = numValues; valIdx > 0; valIdx--) {
        value = values[valIdx - 1].value;

        for (byteIdx = bytesPerVal; byteIdx > 0; byteIdx--) {
            byteNum = bytesPerVal - byteIdx;

            currentByte = (value >> (byteNum * BITS_PER_BYTE)) & 0xFF;
           
            current |= (currentByte << consumedBits);
            current &= 0xFF;

            if ((byteIdx > 2) || (remainderBits == 0)) {
                numBits = BITS_PER_BYTE;
            } else {
                numBits = remainderBits;
            }


            consumedBits += numBits;

            if (consumedBits > BITS_PER_BYTE) {
                consumedBits -= BITS_PER_BYTE;
                
                output[outputIdx] = current;

                usedBits = BITS_PER_BYTE - consumedBits;
                current = currentByte >> usedBits;

            }

        }

    }
    

}

