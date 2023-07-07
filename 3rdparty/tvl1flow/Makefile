CC=c99
CFLAGS=-O3 -I.
OMPFLAGS=-fopenmp

tvl1flow: main.o tvl1flow_lib.o bicubic_interpolation.o mask.o zoom.o iio.o xmalloc.o
	$(CC) $(CFLAGS) $(OMPFLAGS) -o tvl1flow main.o tvl1flow_lib.o bicubic_interpolation.o mask.o zoom.o iio.o xmalloc.o -ltiff -lpng -ljpeg -lm

iio.o: iio.c
	$(CC) $(CFLAGS) -DNDEBUG -D_GNU_SOURCE -c iio.c

clean:
	rm -f *.o tvl1flow
