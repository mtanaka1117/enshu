/*
 * bit_ops.h
 *
 *  Created on: 11 Jun 2021
 *      Author: tejask
 */

#ifndef BIT_OPS_H_
#define BIT_OPS_H_


#define SET_BIT(X, Y)      ((X) |= (Y))
#define CLEAR_BIT(X, Y)    ((X) &= ~(Y))
#define TOGGLE_BIT(X, Y)   ((X) ^= (Y))
#define TEST_BIT(X, Y)     ((X) & (Y))


#endif /* BIT_OPS_H_ */
