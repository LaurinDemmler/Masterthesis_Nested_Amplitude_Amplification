#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <limits.h>  // Changed from values.h
#include <string.h>

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable: 4996)
#endif

/* ======================================================================
                     macros
   ====================================================================== */
#define srand(x)     srand48x(x)
#define randm(x)     (lrand48x() % (x))
#define NO(f,i)      ((int) ((i+1)-f))
#define TRUE  1
#define FALSE 0

/* ======================================================================
                 type declarations
   ====================================================================== */
typedef int   boolean; /* boolean variables */
typedef short itype;   /* item profits and weights */
typedef long  stype;   /* sum of profit or weight */

/* item */
typedef struct {
  itype   p;     /* profit */
  itype   w;     /* weight */
  boolean x;     /* solution variable */
} item;

/* =======================================================================
                                random
   ======================================================================= */
unsigned long _h48, _l48;

void srand48x(long s)
{
  _h48 = s;
  _l48 = 0x330E;
}

long lrand48x(void)
{
  _h48 = (_h48 * 0xDEECE66D) + (_l48 * 0x5DEEC);
  _l48 = _l48 * 0xE66D + 0xB;
  _h48 = _h48 + (_l48 >> 16);
  _l48 = _l48 & 0xFFFF;
  return (_h48 >> 1);
}

/* ======================================================================
                                 error
   ====================================================================== */
void error(char *str, ...)
{
  va_list args;
  va_start(args, str);
  vprintf(str, args); printf("\n");
  va_end(args);
  printf("STOP !!!\n\n"); 
  exit(-1);
}

/* ======================================================================
                  palloc
   ====================================================================== */
void pfree(void *p)
{
  if (p == NULL) error("freeing null");
  free(p);
}

void * palloc(size_t no, size_t each)
{
  long size;
  void *p;  // Changed from long *p
  size = no * (long) each;
  if (size == 0) size = 1;
  if (size != (size_t) size) error("alloc too big %ld", size);
  p = malloc(size);
  if (p == NULL) error("no memory size %ld", size);
  return p;
}

/* ======================================================================
                                showitems
   ====================================================================== */
void showitems(item *f, item *l, stype c)
{
  item *i;
  FILE *out;
 
  out = fopen("test.in", "w"); 
  if (out == NULL) error("no file");
  fprintf(out,"%d\n", NO(f,l));
  for (i = f; i <= l; i++) {
    fprintf(out, "%5d %5d %5d\n", NO(f,i), i->p, i->w);
  }
  fprintf(out,"%ld\n", c);  // Changed %d to %ld for stype
  fclose(out);
}

/* ======================================================================
                maketest
   ====================================================================== */
stype maketest(item *f, item *l, int r, int type, int v, int S)
{
  register item *i;
  register stype sum;
  stype c;
  itype r1;
  srand(v);
  sum = 0; r1 = r/10;
  for (i = f; i <= l; i++) {
    i->w = randm(r) + 1;
    switch (type) {
      case 1: i->p = randm(r) + 1;
          break;
      case 2: i->p = randm(2*r1+1) + i->w - r1;
          if (i->p <= 0) i->p = 1;
          break;
      case 3: i->p = i->w + 10;
          break;
      case 4: i->p = i->w;
          break;
    }
    sum += i->w;
  }
  c = (v * (double) sum) / (S + 1);
  if (c <= r) c = r+1;
  return c;
}

/* ======================================================================
                main
   ====================================================================== */
int main(int argc, char *argv[])
{
  item *f, *l;
  int n, r, type, i, S;
  stype c;
 
  if (argc == 6) {
    n = atoi(argv[1]);
    r = atoi(argv[2]);
    type = atoi(argv[3]);
    i = atoi(argv[4]);
    S = atoi(argv[5]);
    printf("generator %d %d %d %d %d\n", n, r, type, i, S);
  } else {
    printf("generator\n");
    printf("n = ");
    scanf("%d", &n);
    printf("r = ");
    scanf("%d", &r);
    printf("t = ");
    scanf("%d", &type);
    printf("i = ");
    scanf("%d", &i);
    printf("S = ");
    scanf("%d", &S);
  }
  f = palloc(n, sizeof(item));
  l = f + n-1;
  c = maketest(f, l, r, type, i, S); 
  showitems(f, l, c);
  pfree(f);
  return 0;
}