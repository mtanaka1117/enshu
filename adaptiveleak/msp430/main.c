#include <msp430.h> 
#include <stdint.h>

#include "bit_ops.h"
#include "bt_functions.h"
#include "data.h"
#include "init.h"
#include "sampler.h"
#include "policy_parameters.h"
#include "policy.h"
#include "utils/aes256.h"
#include "utils/bitmap.h"
#include "utils/encoding.h"
#include "utils/encryption.h"
#include "utils/matrix.h"
#include "utils/lfsr.h"


#define START_BYTE 0xCC
#define SEND_BYTE  0xBB
#define RESET_BYTE 0xFF

#define START_RESPONSE 0xAB
#define RESET_RESPONSE 0xCD

#define HEADER_SIZE 18
#define OUTPUT_OFFSET 34
#define LENGTH_SIZE 2

// Encryption Key (hard-coded and known to the server)
static const uint8_t AES_KEY[16] = { 52,159,220,0,180,77,26,170,202,163,162,103,15,212,66,68 };
static uint8_t aesIV[16] = { 0x66, 0xa1, 0xfc, 0xdc, 0x34, 0x79, 0x66, 0xee, 0xe4, 0x26, 0xc1, 0x5a, 0x17, 0x9e, 0x78, 0x31 };

// Variable to track the operation mode
enum OpMode { COLLECT, ENCODE, RESET, SEND, START, NOT_STARTED, IDLE };
volatile enum OpMode opMode;

// Counters to track progress
volatile uint16_t seqIdx = 0;
volatile uint16_t elemIdx = 0;

// Buffer for messages to send
static uint8_t outputBuffer[1024];
volatile uint16_t numOutputBytes = 0;
volatile uint16_t numDataBytes = 0;
volatile uint16_t numBytesToSend = 0;

// Buffer of data for feature vectors
static FixedPoint FEATURE_BUFFER[SEQ_LENGTH * NUM_FEATURES];

// Buffer for the collected bitmask
static uint8_t INDEX_BUFFER[BITMASK_BYTES];


// Create buffers for some policies
#ifdef IS_ADAPTIVE_HEURISTIC
static FixedPoint ZERO_BUFFER[NUM_FEATURES];
#elif defined(IS_ADAPTIVE_DEVIATION)
static FixedPoint meanData[NUM_FEATURES];
static FixedPoint devData[NUM_FEATURES];
#elif defined(IS_SKIP_RNN)
static FixedPoint stateData[STATE_SIZE];
#endif


// Create buffers for group encoding policy
#ifdef IS_GROUP_ENCODED
#pragma DATA_SECTION(TEMP_DATA_BUFFER, ".matrix")
static FixedPoint TEMP_DATA_BUFFER[SEQ_LENGTH * NUM_FEATURES];

#pragma DATA_SECTION(SHIFT_BUFFER, ".matrix")
static int8_t SHIFT_BUFFER[SEQ_LENGTH * NUM_FEATURES];

#pragma DATA_SECTION(COUNT_BUFFER, ".matrix")
static uint16_t COUNT_BUFFER[SEQ_LENGTH * NUM_FEATURES];

#endif

/**
 * main.c
 */
int main(void)
{
	WDTCTL = WDTPW | WDTHOLD;	// stop watchdog timer
	
	init_gpio();
	init_uart_pins();
	init_uart_system();
	init_timer();

    // Disable the GPIO power-on default high-impedance mode to activate
    // previously configured port settings
    PM5CTL0 &= ~LOCKLPM5;

    // Set the Encryption Key
    uint8_t status = AES256_setCipherKey(AES256_BASE, AES_KEY, AES256_KEYLENGTH_128BIT);
    if (status == STATUS_FAIL) {
        P1OUT |= BIT0;
        return 1;
    }

    // Initialize the operation mode
    opMode = NOT_STARTED;

    volatile uint8_t shouldCollect = 0;
    volatile uint8_t didCollect = 0;
    volatile uint16_t numCollected = 0;

    // Initialize the feature vector array
    struct Vector featureVectors[SEQ_LENGTH];
    uint16_t i;
    for (i = 0; i < SEQ_LENGTH; i++) {
        featureVectors[i].data = FEATURE_BUFFER + (i * NUM_FEATURES);
        featureVectors[i].size = NUM_FEATURES;
    }

    // Initialize the collected indices bitmap
    struct BitMap collectedIndices = { INDEX_BUFFER, BITMASK_BYTES };

    // Initialize the policy
    #ifdef IS_UNIFORM
    struct UniformPolicy policy;
    uniform_policy_init(&policy, COLLECT_INDICES, NUM_INDICES);

    #elif defined(IS_ADAPTIVE_HEURISTIC)
    struct Vector *prevFeatures;

    // Initialize the zero vector
    struct Vector zeroFeatures = { ZERO_BUFFER, NUM_FEATURES };
    for (i = 0; i < NUM_FEATURES; i++) {
        ZERO_BUFFER[i] = 0;
    }

    struct HeuristicPolicy policy;
    heuristic_policy_init(&policy, MAX_SKIP, MIN_SKIP, THRESHOLD);

    #elif defined(IS_ADAPTIVE_DEVIATION)
    struct DeviationPolicy policy;
    struct Vector mean = { meanData, NUM_FEATURES };
    struct Vector dev = { devData, NUM_FEATURES };

    deviation_policy_init(&policy, MAX_SKIP, MIN_SKIP, THRESHOLD, ALPHA, BETA, &mean, &dev);

    #elif defined(IS_SKIP_RNN)
    struct SkipRNNPolicy policy;
    struct Vector state = { stateData, STATE_SIZE };
    skip_rnn_policy_init(&policy, &W_CANDIDATE, &B_CANDIDATE, &W_UPDATE, &B_UPDATE, &W_STATE, B_STATE, &state, &INITIAL_STATE, &MEAN, &SCALE, RNN_PRECISION);
    #endif

    // Put into Low Power Mode
    __bis_SR_register(LPM3_bits | GIE);

    while (1) {

        if (opMode == START) {
            // Send response to the server
            send_byte(START_RESPONSE);

            // Reset the operation mode
            opMode = IDLE;
        } else if (opMode == RESET) {
            // Reset the policy and collected bitmap
            clear_bitmap(&collectedIndices);

            #ifdef IS_UNIFORM
            uniform_reset(&policy);
            #elif defined(IS_ADAPTIVE_HEURISTIC)
            heuristic_reset(&policy);
            prevFeatures = &zeroFeatures;
            #elif defined(IS_ADAPTIVE_DEVIATION)
            deviation_reset(&policy);
            #elif defined(IS_SKIP_RNN)
            skip_rnn_reset(&policy, &INITIAL_STATE);
            #endif

            // Reset the indices
            seqIdx = 0;
            elemIdx = 0;
            numCollected = 0;

            // Send the response to the server
            send_byte(RESET_RESPONSE);

            // Clear the flags and operation mode
            opMode = NOT_STARTED;
        } else if (opMode == COLLECT) {
            #ifdef IS_UNIFORM
            shouldCollect = uniform_should_collect(&policy, elemIdx);
            #elif defined(IS_ADAPTIVE_HEURISTIC)
            shouldCollect = heuristic_should_collect(&policy, elemIdx);
            #elif defined(IS_ADAPTIVE_DEVIATION)
            shouldCollect = deviation_should_collect(&policy, elemIdx);
            #elif defined(IS_SKIP_RNN)
            shouldCollect = skip_rnn_should_collect(&policy, elemIdx);
            #endif

            if (shouldCollect) {
                // Collect the data
                didCollect = get_measurement((featureVectors + elemIdx)->data, seqIdx, elemIdx, NUM_FEATURES, SEQ_LENGTH);

                if (didCollect) {
                    // Record the collection of this element.
                    set_bit(elemIdx, &collectedIndices);

                    #ifdef IS_ADAPTIVE_HEURISTIC
                    heuristic_update(&policy, featureVectors + elemIdx, prevFeatures);
                    prevFeatures = featureVectors + elemIdx;
                    #elif defined(IS_ADAPTIVE_DEVIATION)
                    deviation_update(&policy, featureVectors + elemIdx, DEFAULT_PRECISION);
                    #elif defined(IS_SKIP_RNN)
                    skip_rnn_update(&policy, featureVectors + elemIdx, DEFAULT_PRECISION);
                    #endif

                    numCollected++;
                }
            }

            // Increment the element index
            elemIdx++;

            // Reset the operation mode
            opMode = IDLE;
        } else if (opMode == ENCODE) {
            // Encode the message
            #ifdef IS_STANDARD_ENCODED
            numDataBytes = encode_standard(outputBuffer + OUTPUT_OFFSET, featureVectors, &collectedIndices, NUM_FEATURES, SEQ_LENGTH);
            #elif defined(IS_GROUP_ENCODED)
            numDataBytes = encode_group(outputBuffer + OUTPUT_OFFSET, featureVectors, &collectedIndices, numCollected, NUM_FEATURES, SEQ_LENGTH, TARGET_BYTES, DEFAULT_PRECISION, MAX_COLLECTED, TEMP_DATA_BUFFER, SHIFT_BUFFER, COUNT_BUFFER, 1);
            #endif

            // Generate the initialization vector via an LFSR step
            lfsr_array(aesIV, AES_BLOCK_SIZE);

            // Encrypt the message
            numOutputBytes = round_to_aes_block(numDataBytes);
            encrypt_aes128(outputBuffer + OUTPUT_OFFSET, aesIV, outputBuffer + HEADER_SIZE, numOutputBytes);

            // Set the IV
            for (i = LENGTH_SIZE; i < HEADER_SIZE; i++) {
                outputBuffer[i] = aesIV[i - LENGTH_SIZE];
            }

            // Set the number of bytes to send
            numBytesToSend = numOutputBytes + HEADER_SIZE;

            // Extend Group Encoded Messages if needed
            #if defined(IS_GROUP_ENCODED) || defined(IS_PADDED)
            if (numBytesToSend < TARGET_BYTES) {
                numBytesToSend = TARGET_BYTES;
            }
            #endif

            // Add the message size for proper retrieval. This leaks no information
            // because recipient can read the message size for themselves
            outputBuffer[0] = (numBytesToSend >> 8) & 0xFF;
            outputBuffer[1] = (numBytesToSend) & 0xFF;

            // Clear the encryption flag and increment the element index to avoid
            // encrypting again
            elemIdx++;
            opMode = IDLE;
        } else if (opMode == SEND) {
            // Send the message to the server
            send_message(outputBuffer, numBytesToSend);

            // Reset the policy and collected bitmap
            clear_bitmap(&collectedIndices);

            #ifdef IS_UNIFORM
            uniform_reset(&policy);
            #elif defined(IS_ADAPTIVE_HEURISTIC)
            heuristic_reset(&policy);
            prevFeatures = &zeroFeatures;
            #elif defined(IS_ADAPTIVE_DEVIATION)
            deviation_reset(&policy);
            #elif defined(IS_SKIP_RNN)
            skip_rnn_reset(&policy, &INITIAL_STATE);
            #endif

            // Reset the indices and move to the next
            // sequence
            seqIdx++;
            elemIdx = 0;
            numCollected = 0;
            numBytesToSend = 0;
            numOutputBytes = 0;

            // Set the operation mode based on whether there
            // are more sequences to capture
            if (seqIdx >= MAX_NUM_SEQ) {
                seqIdx = 0;
            }

            // Reset the operation mode
            opMode = IDLE;
        }

        __bis_SR_register(LPM3_bits | GIE);

    }

	return 0;
}


/**
 * ISR for Timer A overflow
 */
#pragma vector = TIMER0_A1_VECTOR
__interrupt void Timer0_A1_ISR (void) {
    /**
     * Timer Interrupts to make data pull requests.
     */
    switch(__even_in_range(TA0IV, TAIV__TAIFG))
    {
        case TAIV__NONE:   break;           // No interrupt
        case TAIV__TACCR1: break;           // CCR1 not used
        case TAIV__TACCR2: break;           // CCR2 not used
        case TAIV__TACCR3: break;           // reserved
        case TAIV__TACCR4: break;           // reserved
        case TAIV__TACCR5: break;           // reserved
        case TAIV__TACCR6: break;           // reserved
        case TAIV__TAIFG:                   // overflow
            if ((opMode != NOT_STARTED) && (elemIdx <= SEQ_LENGTH)) {
                if (elemIdx < SEQ_LENGTH) {
                    opMode = COLLECT;
                } else {
                    opMode = ENCODE;
                }

                __bic_SR_register_on_exit(LPM3_bits | GIE);
            }

            break;
        default: break;
    }
}

/**
 * ISR for receiving data on UART RX pin.
 */
#pragma vector=EUSCI_A3_VECTOR
__interrupt void USCI_A3_ISR(void) {
    char c;

    switch(__even_in_range(UCA3IV, USCI_UART_UCTXCPTIFG)) {
        case USCI_NONE: break;
        case USCI_UART_UCRXIFG:
            // Wait until TX Buffer is not busy
            while(!(UCA3IFG & UCTXIFG));

            c = (char) UCA3RXBUF;

            if (c == START_BYTE) {
                opMode = START;
                __bic_SR_register_on_exit(LPM3_bits | GIE);
            } else if (c == SEND_BYTE) {
                opMode = SEND;
                __bic_SR_register_on_exit(LPM3_bits | GIE);
            } else if (c == RESET_BYTE) {
                opMode = RESET;
                __bic_SR_register_on_exit(LPM3_bits | GIE);
            }

            break;
        case USCI_UART_UCTXIFG: break;
        case USCI_UART_UCSTTIFG: break;
        case USCI_UART_UCTXCPTIFG: break;
        default: break;
    }
}

