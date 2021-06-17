#include <msp430.h>
#include <stdint.h>

#ifndef BT_FUNCTIONS_H
#define BT_FUNCTIONS_H

// Functions to transmit data
void send_byte(uint8_t c);
void send_message(uint8_t *message, uint16_t numBytes);

#endif
