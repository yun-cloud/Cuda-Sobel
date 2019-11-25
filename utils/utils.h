#include <png.h>

#ifdef __cplusplus
extern "C" {
#endif

int read_png(const char* filename, unsigned char** image, unsigned* height, 
			 unsigned* width, unsigned* channels);


void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width, 
			   const unsigned channels);

#ifdef __cplusplus
}
#endif
