#include <stdint.h>
#include <memory.h>

int32_t sum_and_overflow (int32_t a, int32_t b)
{
  uint32_t au, bu, su;
  int32_t s;
 
  memcpy(&au, &a, sizeof(a));
  memcpy(&bu, &b, sizeof(b));
  su = au + bu;
  memcpy(&s, &su, sizeof(s));
  return s;
  
} 
