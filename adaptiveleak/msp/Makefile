CCBIN=/usr/bin/gcc
CC=$(CCBIN) -Wall -Wextra -std=c99 -pedantic -g -Wno-unused-variable -fsanitize=address

default: policy

policy: utils/constants.h utils/fixed_point.h data.h utils/sampler.h utils/bitmap.h utils/matrix.h utils/encoding.h policy_parameters.h policy.h main.h utils/fixed_point.c utils/bitmap.c utils/matrix.c utils/encoding.c policy.c main.c
	$(CC) -o policy utils/fixed_point.c utils/sampler.c utils/bitmap.c utils/matrix.c utils/encoding.c policy.c main.c

encoding_tests: utils/constants.h utils/fixed_point.h utils/bitmap.h utils/encoding.h tests/encoding_tests.h utils/fixed_point.c utils/bitmap.c utils/encoding.c tests/encoding_tests.c
	$(CC) -o encoding_tests utils/fixed_point.c utils/bitmap.c utils/encoding.c tests/encoding_tests.c

matrix_tests: utils/constants.h utils/fixed_point.h utils/matrix.h tests/matrix_tests.h utils/fixed_point.c utils/matrix.c tests/matrix_tests.c
	$(CC) -o matrix_tests utils/fixed_point.c utils/matrix.c tests/matrix_tests.c
clean: 
	rm -f *.o policy matrix_tests encoding_tests *.dSYM
