/* --COPYRIGHT--,BSD
 * Copyright (c) 2016, Texas Instruments Incorporated
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * *  Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * *  Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * *  Neither the name of Texas Instruments Incorporated nor the names of
 *    its contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * --/COPYRIGHT--*/
//*****************************************************************************
//
// aes256.h - Driver for the AES256 Module.
//
//*****************************************************************************
#include <stdint.h>
#include <stdbool.h>
#include <msp430.h>


#ifndef __MSP430WARE_AES256_H__
#define __MSP430WARE_AES256_H__

//#include "inc/hw_memmap.h"

#define STATUS_SUCCESS  0x01
#define STATUS_FAIL     0x00

#define HWREG32(x)                                                              \
    (*((volatile uint32_t *)((uint16_t)x)))
#define HWREG16(x)                                                             \
    (*((volatile uint16_t *)((uint16_t)x)))
#define HWREG8(x)                                                             \
    (*((volatile uint8_t *)((uint16_t)x)))


#ifdef __MSP430_HAS_AES256__

//*****************************************************************************
//
// If building with a C++ compiler, make all of the definitions in this header
// have a C binding.
//
//*****************************************************************************
#ifdef __cplusplus
extern "C"
{
#endif

//*****************************************************************************
//
// The following are values that can be passed to the keyLength parameter for
// functions: AES256_setCipherKey(), AES256_setDecipherKey(), and
// AES256_startSetDecipherKey().
//
//*****************************************************************************
#define AES256_KEYLENGTH_128BIT                                             128
#define AES256_KEYLENGTH_192BIT                                             192
#define AES256_KEYLENGTH_256BIT                                             256

//*****************************************************************************
//
// The following are values that can be passed toThe following are values that
// can be returned by the AES256_getErrorFlagStatus() function.
//
//*****************************************************************************
#define AES256_ERROR_OCCURRED                                          AESERRFG
#define AES256_NO_ERROR                                                    0x00

//*****************************************************************************
//
// The following are values that can be passed toThe following are values that
// can be returned by the AES256_isBusy() function.
//
//*****************************************************************************
#define AES256_BUSY                                                     AESBUSY
#define AES256_NOT_BUSY                                                    0x00

//*****************************************************************************
//
// The following are values that can be passed toThe following are values that
// can be returned by the AES256_getInterruptStatus() function.
//
//*****************************************************************************
#define AES256_READY_INTERRUPT                                         AESRDYIE
#define AES256_NOTREADY_INTERRUPT                                          0x00

//*****************************************************************************
//
// Prototypes for the APIs.
//
//*****************************************************************************

//*****************************************************************************
//
//! \brief Loads a 128, 192 or 256 bit cipher key to AES256 module.
//!
//! This function loads a 128, 192 or 256 bit cipher key to AES256 module.
//! Requires both a key as well as the length of the key provided. Acceptable
//! key lengths are AES256_KEYLENGTH_128BIT, AES256_KEYLENGTH_192BIT, or
//! AES256_KEYLENGTH_256BIT
//!
//! \param baseAddress is the base address of the AES256 module.
//! \param cipherKey is a pointer to an uint8_t array with a length of 16 bytes
//!        that contains a 128 bit cipher key.
//! \param keyLength is the length of the key.
//!        Valid values are:
//!        - \b AES256_KEYLENGTH_128BIT
//!        - \b AES256_KEYLENGTH_192BIT
//!        - \b AES256_KEYLENGTH_256BIT
//!
//! \return STATUS_SUCCESS or STATUS_FAIL of key loading
//
//*****************************************************************************
extern uint8_t AES256_setCipherKey(uint16_t baseAddress,
                                   const uint8_t *cipherKey,
                                   uint16_t keyLength);

//*****************************************************************************
//
//! \brief Encrypts a block of data using the AES256 module.
//!
//! The cipher key that is used for encryption should be loaded in advance by
//! using function AES256_setCipherKey()
//!
//! \param baseAddress is the base address of the AES256 module.
//! \param data is a pointer to an uint8_t array with a length of 16 bytes that
//!        contains data to be encrypted.
//! \param encryptedData is a pointer to an uint8_t array with a length of 16
//!        bytes in that the encrypted data will be written.
//!
//! \return None
//
//*****************************************************************************
extern void AES256_encryptData(uint16_t baseAddress,
                               const uint8_t *data,
                               const uint8_t *prev,
                               uint8_t *encryptedData);

//*****************************************************************************
//
//! \brief Resets AES256 Module immediately.
//!
//! This function performs a software reset on the AES256 Module, note that
//! this does not affect the AES256 ready interrupt.
//!
//! \param baseAddress is the base address of the AES256 module.
//!
//! Modified bits are \b AESSWRST of \b AESACTL0 register.
//!
//! \return None
//
//*****************************************************************************
extern void AES256_reset(uint16_t baseAddress);

//*****************************************************************************
//
// Mark the end of the C bindings section for C++ compilers.
//
//*****************************************************************************
#ifdef __cplusplus
}
#endif

#endif
#endif // __MSP430WARE_AES256_H__
