CCBIN=/usr/bin/gcc
CC=$(CCBIN) -Wall -Wextra -std=c99 -pedantic -g -Wno-unused-variable -fsanitize=address

default: policy

policy: utils/constants.h utils/fixed_point.h data.h sampler.h utils/bitmap.h utils/matrix.h utils/encoding.h policy_parameters.h policy.h main.h utils/fixed_point.c utils/bitmap.c utils/matrix.c utils/encoding.c policy.c main.c
	$(CC) -o policy utils/fixed_point.c sampler.c utils/bitmap.c utils/matrix.c utils/encoding.c policy.c main.c

encoding_tests: utils/constants.h utils/math.h utils/fixed_point.h utils/bitmap.h utils/matrix.h utils/encryption.h utils/packing.h utils/pruning.h utils/range_shifting.h utils/encoding.h tests/encoding_tests.h utils/math.c utils/fixed_point.c utils/bitmap.c utils/matrix.c utils/encryption.c utils/packing.h utils/pruning.c utils/range_shifting.c utils/encoding.c tests/encoding_tests.c
	$(CC) -o encoding_tests utils/math.c utils/fixed_point.c utils/bitmap.c utils/matrix.c utils/encryption.c utils/packing.c utils/pruning.c utils/range_shifting.c utils/encoding.c tests/encoding_tests.c

matrix_tests: utils/constants.h utils/fixed_point.h utils/matrix.h tests/matrix_tests.h utils/fixed_point.c utils/matrix.c tests/matrix_tests.c
	$(CC) -o matrix_tests utils/fixed_point.c utils/matrix.c tests/matrix_tests.c

packing_tests: utils/constants.h utils/fixed_point.h utils/packing.h tests/packing_tests.h utils/fixed_point.c utils/packing.c tests/packing_tests.c
	$(CC) -o packing_tests utils/fixed_point.c utils/packing.c tests/packing_tests.c

range_tests: utils/constants.h utils/math.h utils/fixed_point.h utils/range_shifting.h tests/range_shifting_tests.h utils/math.c utils/fixed_point.c utils/range_shifting.c tests/range_shifting_tests.c
	$(CC) -o range_tests utils/math.c utils/fixed_point.c utils/range_shifting.c tests/range_shifting_tests.c

clean: 
	rm -f *.o policy matrix_tests encoding_tests packing_tests range_tests *.dSYM
