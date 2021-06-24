/*
 * init.h
 *
 *  Created on: 11 Jun 2021
 *      Author: tejask
 */
#include <msp430.h>
#include "bit_ops.h"

#ifndef INIT_H_
#define INIT_H_

void init_gpio(void);
void init_uart_pins(void);
void init_uart_system(void);
void init_timer(void);


#endif /* INIT_H_ */
