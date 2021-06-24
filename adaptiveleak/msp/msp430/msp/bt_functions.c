#include "bt_functions.h"

// Functions to transmit data

void send_byte(uint8_t c) {
    // Wait until the Tx Buffer is free
    while(!(UCA3IFG & UCTXIFG));
    UCA3TXBUF = c;

    // Wait until byte has been sent.
    while(UCA3STATW & UCBUSY);
}

void send_message(uint8_t *message, uint16_t numBytes) {
    uint16_t i;
    for (i = 0; i < numBytes; i++) {
        send_byte(message[i]);
    }
}
