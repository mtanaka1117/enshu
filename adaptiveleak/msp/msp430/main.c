#include <msp430.h> 
#include <stdint.h>

#include "bit_ops.h"
#include "data.h"
#include "init.h"
#include "sampler.h"
#include "policy_parameters.h"
#include "policy.h"
#include "utils/encoding.h"
#include "utils/bitmap.h"
#include "utils/matrix.h"
#include "bt_functions.h"


#define START_BYTE 0xFE
#define SEND_BYTE  0xEE
#define RESET_BYTE 0xFF

volatile uint8_t shouldSend = 0; // Denotes when the system should send data.
volatile uint8_t shouldSample = 0;
volatile uint8_t shouldReset = 0;
volatile uint8_t isStarted = 0;
volatile uint16_t seqIdx = 0;
volatile uint16_t elemIdx = 0;
volatile uint16_t numCollected = 0;

// Buffer for messages to send
static uint8_t outputBuffer[800];
volatile uint16_t numOutputBytes = 0;

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

	CLEAR_BIT(P1OUT, BIT0);

    // Disable the GPIO power-on default high-impedance mode to activate
    // previously configured port settings
    PM5CTL0 &= ~LOCKLPM5;

    seqIdx = 0;
    elemIdx = 0;
    numOutputBytes = 0;

    uint8_t shouldCollect = 0;
    uint8_t didCollect = 0;

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
    heuristic_policy_init(&policy, MAX_SKIP, THRESHOLD);

    #elif defined(IS_ADAPTIVE_DEVIATION)
    struct DeviationPolicy policy;
    struct Vector mean = { meanData, NUM_FEATURES };
    struct Vector dev = { devData, NUM_FEATURES };

    deviation_policy_init(&policy, MAX_SKIP, THRESHOLD, ALPHA, BETA, &mean, &dev);

    #elif defined(IS_SKIP_RNN)
    struct SkipRNNPolicy policy;
    struct Vector state = { stateData, STATE_SIZE };
    skip_rnn_policy_init(&policy, &W_CANDIDATE, &B_CANDIDATE, &W_UPDATE, &B_UPDATE, &W_STATE, B_STATE, &state, &INITIAL_STATE, &MEAN, &SCALE, RNN_PRECISION);
    #endif

    // Put into Low Power Mode
    __bis_SR_register(LPM0_bits | GIE);

    while (1) {
        if (shouldSample) {

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

            elemIdx++;  // Increment the element index

            shouldSample = 0;
        } else if (shouldSend) {
            // Encode the message
            numOutputBytes = encode_standard(outputBuffer, featureVectors, &collectedIndices, NUM_FEATURES, SEQ_LENGTH);

            // Send the message to the server
            send_message(outputBuffer, numOutputBytes);

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
            seqIdx++;
            elemIdx = 0;
            numCollected = 0;

            if (seqIdx >= MAX_NUM_SEQ) {
                seqIdx = 0;
            }

            // Clear the sending flag
            shouldSend = 0;
        } else if (shouldReset) {
            // Clear the flags
            shouldSend = 0;
            shouldSample = 0;
            shouldReset = 0;
            isStarted = 0;

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
        }

        __bis_SR_register(LPM0_bits | GIE);

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
            if (isStarted && (elemIdx < SEQ_LENGTH)) {
                shouldSample = 1;
                __bic_SR_register_on_exit(LPM0_bits | GIE);
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
                isStarted = 1;
            } else if (c == SEND_BYTE) {
                shouldSend = 1;
                __bic_SR_register_on_exit(LPM0_bits | GIE);
            } else if (c == RESET_BYTE) {
                shouldReset = 1;
                __bic_SR_register_on_exit(LPM0_bits | GIE);
            }

            break;
        case USCI_UART_UCTXIFG: break;
        case USCI_UART_UCSTTIFG: break;
        case USCI_UART_UCTXCPTIFG: break;
        default: break;
    }
}

