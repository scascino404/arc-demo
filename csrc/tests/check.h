/*
 * check.h - minimal test reporting shared by the C test programs.
 */
#ifndef ARC_CHECK_H
#define ARC_CHECK_H

#include <stdio.h>
#include "../arc_math.h"

#define DT (1.0 / 30.0)
#define SUBSTEPS 20

static int n_fail = 0, n_pass = 0;

/* ARC_INLINE, so programs that skip a helper get no unused-function warning */

ARC_INLINE void check(int ok, const char *name, double got, double expected)
{
    if (ok) ++n_pass; else ++n_fail;
    printf("  %-5s %-44s got % .6f (expected % .6f)\n", ok ? "ok" : "FAIL", name, got, expected);
}

ARC_INLINE void check_near(const char *name, double got, double expected, double tol)
{
    check(fabs(got - expected) <= tol, name, got, expected);
}

ARC_INLINE void check_below(const char *name, double got, double limit)
{
    check(got <= limit, name, got, limit);
}

/* Prints the totals; returns the exit status. */
ARC_INLINE int check_summary(void)
{
    printf("\n%d passed, %d failed\n", n_pass, n_fail);
    return n_fail ? 1 : 0;
}

#endif
