/*++

Module Name:

    _pow.c

Abstract:

    This module contains an implementation of `double pow(double, double)` from
    http://www.netlib.org/fdlibm, plus supporting routines (sqrt, copysign, and
    scalbn).  (The fact that we need to do this is a good example of one of the
    downsides of not depending on the standard C runtime.)

--*/

#include "stdafx.h"

#define __STDC__

#define __HI(x) *(1+(int*)&x)
#define __LO(x) *(int*)&x
#define __HIp(x) *(1+(int*)x)
#define __LOp(x) *(int*)x
#define __P(p)  p

#ifdef __STDC__
static const double
#else
static double
#endif
two54   =  1.80143985094819840000e+16, /* 0x43500000, 0x00000000 */
twom54  =  5.55111512312578270212e-17, /* 0x3C900000, 0x00000000 */
one    = 1.0,
huge   = 1.0e+300,
tiny   = 1.0e-300;

//
// As of VS 2022, this is required to prevent the following:
//  "'fabs': intrinsic function not declared
//

double fabs(double);
#pragma intrinsic(fabs)

//
// Begin e_sqrt.c
//

/* @(#)e_sqrt.c 1.3 95/01/18 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

#define sqrt __ieee754_sqrt

#ifdef __STDC__
        double __ieee754_sqrt(double x)
#else
        double __ieee754_sqrt(x)
        double x;
#endif
{
        double z;
        int     sign = (int)0x80000000;
        unsigned r,t1,s1,ix1,q1;
        int ix0,s0,q,m,t,i;

        ix0 = __HI(x);          /* high word of x */
        ix1 = __LO(x);          /* low word of x */

    /* take care of Inf and NaN */
        if((ix0&0x7ff00000)==0x7ff00000) {
            return x*x+x;               /* sqrt(NaN)=NaN, sqrt(+inf)=+inf
                                           sqrt(-inf)=sNaN */
        }
    /* take care of zero */
        if(ix0<=0) {
            if(((ix0&(~sign))|ix1)==0) return x;/* sqrt(+-0) = +-0 */
            else if(ix0<0)
                return (x-x)/(x-x);             /* sqrt(-ve) = sNaN */
        }
    /* normalize x */
        m = (ix0>>20);
        if(m==0) {                              /* subnormal x */
            while(ix0==0) {
                m -= 21;
                ix0 |= (ix1>>11); ix1 <<= 21;
            }
            for(i=0;(ix0&0x00100000)==0;i++) ix0<<=1;
            m -= i-1;
            ix0 |= (ix1>>(32-i));
            ix1 <<= i;
        }
        m -= 1023;      /* unbias exponent */
        ix0 = (ix0&0x000fffff)|0x00100000;
        if(m&1){        /* odd m, double x to make it even */
            ix0 += ix0 + ((ix1&sign)>>31);
            ix1 += ix1;
        }
        m >>= 1;        /* m = [m/2] */

    /* generate sqrt(x) bit by bit */
        ix0 += ix0 + ((ix1&sign)>>31);
        ix1 += ix1;
        q = q1 = s0 = s1 = 0;   /* [q,q1] = sqrt(x) */
        r = 0x00200000;         /* r = moving bit from right to left */

        while(r!=0) {
            t = s0+r;
            if(t<=ix0) {
                s0   = t+r;
                ix0 -= t;
                q   += r;
            }
            ix0 += ix0 + ((ix1&sign)>>31);
            ix1 += ix1;
            r>>=1;
        }

        r = sign;
        while(r!=0) {
            t1 = s1+r;
            t  = s0;
            if((t<ix0)||((t==ix0)&&(t1<=ix1))) {
                s1  = t1+r;
                if(((t1&sign)==(unsigned)sign)&&(s1&sign)==0) s0 += 1;
                ix0 -= t;
                if (ix1 < t1) ix0 -= 1;
                ix1 -= t1;
                q1  += r;
            }
            ix0 += ix0 + ((ix1&sign)>>31);
            ix1 += ix1;
            r>>=1;
        }

    /* use floating add to find out rounding direction */
        if((ix0|ix1)!=0) {
            z = one-tiny; /* trigger inexact flag */
            if (z>=one) {
                z = one+tiny;
                if (q1==(unsigned)0xffffffff) { q1=0; q += 1;}
                else if (z>one) {
                    if (q1==(unsigned)0xfffffffe) q+=1;
                    q1+=2;
                } else
                    q1 += (q1&1);
            }
        }
        ix0 = (q>>1)+0x3fe00000;
        ix1 =  q1>>1;
        if ((q&1)==1) ix1 |= sign;
        ix0 += (m <<20);
        __HI(z) = ix0;
        __LO(z) = ix1;
        return z;
}

/*
Other methods  (use floating-point arithmetic)
-------------
(This is a copy of a drafted paper by Prof W. Kahan
and K.C. Ng, written in May, 1986)

        Two algorithms are given here to implement sqrt(x)
        (IEEE double precision arithmetic) in software.
        Both supply sqrt(x) correctly rounded. The first algorithm (in
        Section A) uses newton iterations and involves four divisions.
        The second one uses reciproot iterations to avoid division, but
        requires more multiplications. Both algorithms need the ability
        to chop results of arithmetic operations instead of round them,
        and the INEXACT flag to indicate when an arithmetic operation
        is executed exactly with no roundoff error, all part of the
        standard (IEEE 754-1985). The ability to perform shift, add,
        subtract and logical AND operations upon 32-bit words is needed
        too, though not part of the standard.

A.  sqrt(x) by Newton Iteration

   (1)  Initial approximation

        Let x0 and x1 be the leading and the trailing 32-bit words of
        a floating point number x (in IEEE double format) respectively

            1    11                  52                           ...widths
           ------------------------------------------------------
        x: |s|    e     |             f                         |
           ------------------------------------------------------
              msb    lsb  msb                                 lsb ...order


             ------------------------        ------------------------
        x0:  |s|   e    |    f1     |    x1: |          f2           |
             ------------------------        ------------------------

        By performing shifts and subtracts on x0 and x1 (both regarded
        as integers), we obtain an 8-bit approximation of sqrt(x) as
        follows.

                k  := (x0>>1) + 0x1ff80000;
                y0 := k - T1[31&(k>>15)].       ... y ~ sqrt(x) to 8 bits
        Here k is a 32-bit integer and T1[] is an integer array containing
        correction terms. Now magically the floating value of y (y's
        leading 32-bit word is y0, the value of its trailing word is 0)
        approximates sqrt(x) to almost 8-bit.

        Value of T1:
        static int T1[32]= {
        0,      1024,   3062,   5746,   9193,   13348,  18162,  23592,
        29598,  36145,  43202,  50740,  58733,  67158,  75992,  85215,
        83599,  71378,  60428,  50647,  41945,  34246,  27478,  21581,
        16499,  12183,  8588,   5674,   3403,   1742,   661,    130,};

    (2) Iterative refinement

        Apply Heron's rule three times to y, we have y approximates
        sqrt(x) to within 1 ulp (Unit in the Last Place):

                y := (y+x/y)/2          ... almost 17 sig. bits
                y := (y+x/y)/2          ... almost 35 sig. bits
                y := y-(y-x/y)/2        ... within 1 ulp


        Remark 1.
            Another way to improve y to within 1 ulp is:

                y := (y+x/y)            ... almost 17 sig. bits to 2*sqrt(x)
                y := y - 0x00100006     ... almost 18 sig. bits to sqrt(x)

                                2
                            (x-y )*y
                y := y + 2* ----------  ...within 1 ulp
                               2
                             3y  + x


        This formula has one division fewer than the one above; however,
        it requires more multiplications and additions. Also x must be
        scaled in advance to avoid spurious overflow in evaluating the
        expression 3y*y+x. Hence it is not recommended uless division
        is slow. If division is very slow, then one should use the
        reciproot algorithm given in section B.

    (3) Final adjustment

        By twiddling y's last bit it is possible to force y to be
        correctly rounded according to the prevailing rounding mode
        as follows. Let r and i be copies of the rounding mode and
        inexact flag before entering the square root program. Also we
        use the expression y+-ulp for the next representable floating
        numbers (up and down) of y. Note that y+-ulp = either fixed
        point y+-1, or multiply y by nextafter(1,+-inf) in chopped
        mode.

                I := FALSE;     ... reset INEXACT flag I
                R := RZ;        ... set rounding mode to round-toward-zero
                z := x/y;       ... chopped quotient, possibly inexact
                If(not I) then {        ... if the quotient is exact
                    if(z=y) {
                        I := i;  ... restore inexact flag
                        R := r;  ... restore rounded mode
                        return sqrt(x):=y.
                    } else {
                        z := z - ulp;   ... special rounding
                    }
                }
                i := TRUE;              ... sqrt(x) is inexact
                If (r=RN) then z=z+ulp  ... rounded-to-nearest
                If (r=RP) then {        ... round-toward-+inf
                    y = y+ulp; z=z+ulp;
                }
                y := y+z;               ... chopped sum
                y0:=y0-0x00100000;      ... y := y/2 is correctly rounded.
                I := i;                 ... restore inexact flag
                R := r;                 ... restore rounded mode
                return sqrt(x):=y.

    (4) Special cases

        Square root of +inf, +-0, or NaN is itself;
        Square root of a negative number is NaN with invalid signal.


B.  sqrt(x) by Reciproot Iteration

   (1)  Initial approximation

        Let x0 and x1 be the leading and the trailing 32-bit words of
        a floating point number x (in IEEE double format) respectively
        (see section A). By performing shifs and subtracts on x0 and y0,
        we obtain a 7.8-bit approximation of 1/sqrt(x) as follows.

            k := 0x5fe80000 - (x0>>1);
            y0:= k - T2[63&(k>>14)].    ... y ~ 1/sqrt(x) to 7.8 bits

        Here k is a 32-bit integer and T2[] is an integer array
        containing correction terms. Now magically the floating
        value of y (y's leading 32-bit word is y0, the value of
        its trailing word y1 is set to zero) approximates 1/sqrt(x)
        to almost 7.8-bit.

        Value of T2:
        static int T2[64]= {
        0x1500, 0x2ef8, 0x4d67, 0x6b02, 0x87be, 0xa395, 0xbe7a, 0xd866,
        0xf14a, 0x1091b,0x11fcd,0x13552,0x14999,0x15c98,0x16e34,0x17e5f,
        0x18d03,0x19a01,0x1a545,0x1ae8a,0x1b5c4,0x1bb01,0x1bfde,0x1c28d,
        0x1c2de,0x1c0db,0x1ba73,0x1b11c,0x1a4b5,0x1953d,0x18266,0x16be0,
        0x1683e,0x179d8,0x18a4d,0x19992,0x1a789,0x1b445,0x1bf61,0x1c989,
        0x1d16d,0x1d77b,0x1dddf,0x1e2ad,0x1e5bf,0x1e6e8,0x1e654,0x1e3cd,
        0x1df2a,0x1d635,0x1cb16,0x1be2c,0x1ae4e,0x19bde,0x1868e,0x16e2e,
        0x1527f,0x1334a,0x11051,0xe951, 0xbe01, 0x8e0d, 0x5924, 0x1edd,};

    (2) Iterative refinement

        Apply Reciproot iteration three times to y and multiply the
        result by x to get an approximation z that matches sqrt(x)
        to about 1 ulp. To be exact, we will have
                -1ulp < sqrt(x)-z<1.0625ulp.

        ... set rounding mode to Round-to-nearest
           y := y*(1.5-0.5*x*y*y)       ... almost 15 sig. bits to 1/sqrt(x)
           y := y*((1.5-2^-30)+0.5*x*y*y)... about 29 sig. bits to 1/sqrt(x)
        ... special arrangement for better accuracy
           z := x*y                     ... 29 bits to sqrt(x), with z*y<1
           z := z + 0.5*z*(1-z*y)       ... about 1 ulp to sqrt(x)

        Remark 2. The constant 1.5-2^-30 is chosen to bias the error so that
        (a) the term z*y in the final iteration is always less than 1;
        (b) the error in the final result is biased upward so that
                -1 ulp < sqrt(x) - z < 1.0625 ulp
            instead of |sqrt(x)-z|<1.03125ulp.

    (3) Final adjustment

        By twiddling y's last bit it is possible to force y to be
        correctly rounded according to the prevailing rounding mode
        as follows. Let r and i be copies of the rounding mode and
        inexact flag before entering the square root program. Also we
        use the expression y+-ulp for the next representable floating
        numbers (up and down) of y. Note that y+-ulp = either fixed
        point y+-1, or multiply y by nextafter(1,+-inf) in chopped
        mode.

        R := RZ;                ... set rounding mode to round-toward-zero
        switch(r) {
            case RN:            ... round-to-nearest
               if(x<= z*(z-ulp)...chopped) z = z - ulp; else
               if(x<= z*(z+ulp)...chopped) z = z; else z = z+ulp;
               break;
            case RZ:case RM:    ... round-to-zero or round-to--inf
               R:=RP;           ... reset rounding mod to round-to-+inf
               if(x<z*z ... rounded up) z = z - ulp; else
               if(x>=(z+ulp)*(z+ulp) ...rounded up) z = z+ulp;
               break;
            case RP:            ... round-to-+inf
               if(x>(z+ulp)*(z+ulp)...chopped) z = z+2*ulp; else
               if(x>z*z ...chopped) z = z+ulp;
               break;
        }

        Remark 3. The above comparisons can be done in fixed point. For
        example, to compare x and w=z*z chopped, it suffices to compare
        x1 and w1 (the trailing parts of x and w), regarding them as
        two's complement integers.

        ...Is z an exact square root?
        To determine whether z is an exact square root of x, let z1 be the
        trailing part of z, and also let x0 and x1 be the leading and
        trailing parts of x.

        If ((z1&0x03ffffff)!=0) ... not exact if trailing 26 bits of z!=0
            I := 1;             ... Raise Inexact flag: z is not exact
        else {
            j := 1 - [(x0>>20)&1]       ... j = logb(x) mod 2
            k := z1 >> 26;              ... get z's 25-th and 26-th
                                            fraction bits
            I := i or (k&j) or ((k&(j+j+1))!=(x1&3));
        }
        R:= r           ... restore rounded mode
        return sqrt(x):=z.

        If multiplication is cheaper then the foregoing red tape, the
        Inexact flag can be evaluated by

            I := i;
            I := (z*z!=x) or I.

        Note that z*z can overwrite I; this value must be sensed if it is
        True.

        Remark 4. If z*z = x exactly, then bit 25 to bit 0 of z1 must be
        zero.

                    --------------------
                z1: |        f2        |
                    --------------------
                bit 31             bit 0

        Further more, bit 27 and 26 of z1, bit 0 and 1 of x1, and the odd
        or even of logb(x) have the following relations:

        -------------------------------------------------
        bit 27,26 of z1         bit 1,0 of x1   logb(x)
        -------------------------------------------------
        00                      00              odd and even
        01                      01              even
        10                      10              odd
        10                      00              even
        11                      01              even
        -------------------------------------------------

    (4) Special cases (see (4) of Section A).

 */


//
// Begin s_copysign.c.
//

/* @(#)s_copysign.c 1.3 95/01/18 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

/*
 * copysign(double x, double y)
 * copysign(x,y) returns a value with the magnitude of x and
 * with the sign bit of y.
 */

#ifdef __STDC__
        double copysign(double x, double y)
#else
        double copysign(x,y)
        double x,y;
#endif
{
        __HI(x) = (__HI(x)&0x7fffffff)|(__HI(y)&0x80000000);
        return x;
}

//
// Begin s_scalbn.c.
//

/* @(#)s_scalbn.c 1.3 95/01/18 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

/*
 * scalbn (double x, int n)
 * scalbn(x,n) returns x* 2**n  computed by  exponent
 * manipulation rather than by actually performing an
 * exponentiation or a multiplication.
 */

//#include "fdlibm.h"

#ifdef __STDC__
        double scalbn (double x, int n)
#else
        double scalbn (x,n)
        double x; int n;
#endif
{
        int  k,hx,lx;
        hx = __HI(x);
        lx = __LO(x);
        k = (hx&0x7ff00000)>>20;                /* extract exponent */
        if (k==0) {                             /* 0 or subnormal x */
            if ((lx|(hx&0x7fffffff))==0) return x; /* +-0 */
            x *= two54;
            hx = __HI(x);
            k = ((hx&0x7ff00000)>>20) - 54;
            if (n< -50000) return tiny*x;       /*underflow*/
            }
        if (k==0x7ff) return x+x;               /* NaN or Inf */
        k = k+n;
        if (k >  0x7fe) return huge*copysign(huge,x); /* overflow  */
        if (k > 0)                              /* normal result */
            {__HI(x) = (hx&0x800fffff)|(k<<20); return x;}
        if (k <= -54)
            if (n > 50000)      /* in case integer overflow in n+k */
                return huge*copysign(huge,x);   /*overflow*/
            else return tiny*copysign(tiny,x);  /*underflow*/
        k += 54;                                /* subnormal result */
        __HI(x) = (hx&0x800fffff)|(k<<20);
        return x*twom54;
}

//
// Begin e_pow.c.
//

/*
 * ====================================================
 * Copyright (C) 2004 by Sun Microsystems, Inc. All rights reserved.
 *
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

/* __ieee754_pow(x,y) return x**y
 *
 *                    n
 * Method:  Let x =  2   * (1+f)
 *      1. Compute and return log2(x) in two pieces:
 *              log2(x) = w1 + w2,
 *         where w1 has 53-24 = 29 bit trailing zeros.
 *      2. Perform y*log2(x) = n+y' by simulating muti-precision
 *         arithmetic, where |y'|<=0.5.
 *      3. Return x**y = 2**n*exp(y'*log2)
 *
 * Special cases:
 *      1.  (anything) ** 0  is 1
 *      2.  (anything) ** 1  is itself
 *      3.  (anything) ** NAN is NAN
 *      4.  NAN ** (anything except 0) is NAN
 *      5.  +-(|x| > 1) **  +INF is +INF
 *      6.  +-(|x| > 1) **  -INF is +0
 *      7.  +-(|x| < 1) **  +INF is +0
 *      8.  +-(|x| < 1) **  -INF is +INF
 *      9.  +-1         ** +-INF is NAN
 *      10. +0 ** (+anything except 0, NAN)               is +0
 *      11. -0 ** (+anything except 0, NAN, odd integer)  is +0
 *      12. +0 ** (-anything except 0, NAN)               is +INF
 *      13. -0 ** (-anything except 0, NAN, odd integer)  is +INF
 *      14. -0 ** (odd integer) = -( +0 ** (odd integer) )
 *      15. +INF ** (+anything except 0,NAN) is +INF
 *      16. +INF ** (-anything except 0,NAN) is +0
 *      17. -INF ** (anything)  = -0 ** (-anything)
 *      18. (-anything) ** (integer) is (-1)**(integer)*(+anything**integer)
 *      19. (-anything except 0 and inf) ** (non-integer) is NAN
 *
 * Accuracy:
 *      pow(x,y) returns x**y nearly rounded. In particular
 *                      pow(integer,integer)
 *      always returns the correct integer provided it is
 *      representable.
 *
 * Constants :
 * The hexadecimal values are the intended ones for the following
 * constants. The decimal values may be used, provided that the
 * compiler will convert from decimal to binary accurately enough
 * to produce the hexadecimal values shown.
 */

//#include "fdlibm.h"

#ifdef __STDC__
static const double
#else
static double
#endif
bp[] = {1.0, 1.5,},
dp_h[] = { 0.0, 5.84962487220764160156e-01,}, /* 0x3FE2B803, 0x40000000 */
dp_l[] = { 0.0, 1.35003920212974897128e-08,}, /* 0x3E4CFDEB, 0x43CFD006 */
zero    =  0.0,
two     =  2.0,
two53   =  9007199254740992.0,  /* 0x43400000, 0x00000000 */
        /* poly coefs for (3/2)*(log(x)-2s-2/3*s**3 */
L1  =  5.99999999999994648725e-01, /* 0x3FE33333, 0x33333303 */
L2  =  4.28571428578550184252e-01, /* 0x3FDB6DB6, 0xDB6FABFF */
L3  =  3.33333329818377432918e-01, /* 0x3FD55555, 0x518F264D */
L4  =  2.72728123808534006489e-01, /* 0x3FD17460, 0xA91D4101 */
L5  =  2.30660745775561754067e-01, /* 0x3FCD864A, 0x93C9DB65 */
L6  =  2.06975017800338417784e-01, /* 0x3FCA7E28, 0x4A454EEF */
P1   =  1.66666666666666019037e-01, /* 0x3FC55555, 0x5555553E */
P2   = -2.77777777770155933842e-03, /* 0xBF66C16C, 0x16BEBD93 */
P3   =  6.61375632143793436117e-05, /* 0x3F11566A, 0xAF25DE2C */
P4   = -1.65339022054652515390e-06, /* 0xBEBBBD41, 0xC5D26BF1 */
P5   =  4.13813679705723846039e-08, /* 0x3E663769, 0x72BEA4D0 */
lg2  =  6.93147180559945286227e-01, /* 0x3FE62E42, 0xFEFA39EF */
lg2_h  =  6.93147182464599609375e-01, /* 0x3FE62E43, 0x00000000 */
lg2_l  = -1.90465429995776804525e-09, /* 0xBE205C61, 0x0CA86C39 */
ovt =  8.0085662595372944372e-0017, /* -(1024-log2(ovfl+.5ulp)) */
cp    =  9.61796693925975554329e-01, /* 0x3FEEC709, 0xDC3A03FD =2/(3ln2) */
cp_h  =  9.61796700954437255859e-01, /* 0x3FEEC709, 0xE0000000 =(float)cp */
cp_l  = -7.02846165095275826516e-09, /* 0xBE3E2FE0, 0x145B01F5 =tail of cp_h*/
ivln2    =  1.44269504088896338700e+00, /* 0x3FF71547, 0x652B82FE =1/ln2 */
ivln2_h  =  1.44269502162933349609e+00, /* 0x3FF71547, 0x60000000 =24b 1/ln2*/
ivln2_l  =  1.92596299112661746887e-08; /* 0x3E54AE0B, 0xF85DDF44 =1/ln2 tail*/

#ifdef __STDC__
        double __ieee754_pow(double x, double y)
#else
        double __ieee754_pow(x,y)
        double x, y;
#endif
{
        double z,ax,z_h,z_l,p_h,p_l;
        double y1,t1,t2,r,s,t,u,v,w;
        int i0,i1,i,j,k,yisint,n;
        int hx,hy,ix,iy;
        unsigned lx,ly;

        i0 = ((*(int*)&one)>>29)^1; i1=1-i0;
        hx = __HI(x); lx = __LO(x);
        hy = __HI(y); ly = __LO(y);
        ix = hx&0x7fffffff;  iy = hy&0x7fffffff;

    /* y==zero: x**0 = 1 */
        if((iy|ly)==0) return one;

    /* +-NaN return x+y */
        if(ix > 0x7ff00000 || ((ix==0x7ff00000)&&(lx!=0)) ||
           iy > 0x7ff00000 || ((iy==0x7ff00000)&&(ly!=0)))
                return x+y;

    /* determine if y is an odd int when x < 0
     * yisint = 0       ... y is not an integer
     * yisint = 1       ... y is an odd int
     * yisint = 2       ... y is an even int
     */
        yisint  = 0;
        if(hx<0) {
            if(iy>=0x43400000) yisint = 2; /* even integer y */
            else if(iy>=0x3ff00000) {
                k = (iy>>20)-0x3ff;        /* exponent */
                if(k>20) {
                    j = ly>>(52-k);
                    if((unsigned)(j<<(52-k))==ly) yisint = 2-(j&1);
                } else if(ly==0) {
                    j = iy>>(20-k);
                    if((j<<(20-k))==iy) yisint = 2-(j&1);
                }
            }
        }

    /* special value of y */
        if(ly==0) {
            if (iy==0x7ff00000) {       /* y is +-inf */
                if(((ix-0x3ff00000)|lx)==0)
                    return  y - y;      /* inf**+-1 is NaN */
                else if (ix >= 0x3ff00000)/* (|x|>1)**+-inf = inf,0 */
                    return (hy>=0)? y: zero;
                else                    /* (|x|<1)**-,+inf = inf,0 */
                    return (hy<0)?-y: zero;
            }
            if(iy==0x3ff00000) {        /* y is  +-1 */
                if(hy<0) return one/x; else return x;
            }
            if(hy==0x40000000) return x*x; /* y is  2 */
            if(hy==0x3fe00000) {        /* y is  0.5 */
                if(hx>=0)       /* x >= +0 */
                return sqrt(x);
            }
        }

        ax   = fabs(x);
    /* special value of x */
        if(lx==0) {
            if(ix==0x7ff00000||ix==0||ix==0x3ff00000){
                z = ax;                 /*x is +-0,+-inf,+-1*/
                if(hy<0) z = one/z;     /* z = (1/|x|) */
                if(hx<0) {
                    if(((ix-0x3ff00000)|yisint)==0) {
                        z = (z-z)/(z-z); /* (-1)**non-int is NaN */
                    } else if(yisint==1)
                        z = -z;         /* (x<0)**odd = -(|x|**odd) */
                }
                return z;
            }
        }

        n = (hx>>31)+1;

    /* (x<0)**(non-int) is NaN */
        if((n|yisint)==0) return (x-x)/(x-x);

        s = one; /* s (sign of result -ve**odd) = -1 else = 1 */
        if((n|(yisint-1))==0) s = -one;/* (-ve)**(odd int) */

    /* |y| is huge */
        if(iy>0x41e00000) { /* if |y| > 2**31 */
            if(iy>0x43f00000){  /* if |y| > 2**64, must o/uflow */
                if(ix<=0x3fefffff) return (hy<0)? huge*huge:tiny*tiny;
                if(ix>=0x3ff00000) return (hy>0)? huge*huge:tiny*tiny;
            }
        /* over/underflow if x is not close to one */
            if(ix<0x3fefffff) return (hy<0)? s*huge*huge:s*tiny*tiny;
            if(ix>0x3ff00000) return (hy>0)? s*huge*huge:s*tiny*tiny;
        /* now |1-x| is tiny <= 2**-20, suffice to compute
           log(x) by x-x^2/2+x^3/3-x^4/4 */
            t = ax-one;         /* t has 20 trailing zeros */
            w = (t*t)*(0.5-t*(0.3333333333333333333333-t*0.25));
            u = ivln2_h*t;      /* ivln2_h has 21 sig. bits */
            v = t*ivln2_l-w*ivln2;
            t1 = u+v;
            __LO(t1) = 0;
            t2 = v-(t1-u);
        } else {
            double ss,s2,s_h,s_l,t_h,t_l;
            n = 0;
        /* take care subnormal number */
            if(ix<0x00100000)
                {ax *= two53; n -= 53; ix = __HI(ax); }
            n  += ((ix)>>20)-0x3ff;
            j  = ix&0x000fffff;
        /* determine interval */
            ix = j|0x3ff00000;          /* normalize ix */
            if(j<=0x3988E) k=0;         /* |x|<sqrt(3/2) */
            else if(j<0xBB67A) k=1;     /* |x|<sqrt(3)   */
            else {k=0;n+=1;ix -= 0x00100000;}
            __HI(ax) = ix;

        /* compute ss = s_h+s_l = (x-1)/(x+1) or (x-1.5)/(x+1.5) */
            u = ax-bp[k];               /* bp[0]=1.0, bp[1]=1.5 */
            v = one/(ax+bp[k]);
            ss = u*v;
            s_h = ss;
            __LO(s_h) = 0;
        /* t_h=ax+bp[k] High */
            t_h = zero;
            __HI(t_h)=((ix>>1)|0x20000000)+0x00080000+(k<<18);
            t_l = ax - (t_h-bp[k]);
            s_l = v*((u-s_h*t_h)-s_h*t_l);
        /* compute log(ax) */
            s2 = ss*ss;
            r = s2*s2*(L1+s2*(L2+s2*(L3+s2*(L4+s2*(L5+s2*L6)))));
            r += s_l*(s_h+ss);
            s2  = s_h*s_h;
            t_h = 3.0+s2+r;
            __LO(t_h) = 0;
            t_l = r-((t_h-3.0)-s2);
        /* u+v = ss*(1+...) */
            u = s_h*t_h;
            v = s_l*t_h+t_l*ss;
        /* 2/(3log2)*(ss+...) */
            p_h = u+v;
            __LO(p_h) = 0;
            p_l = v-(p_h-u);
            z_h = cp_h*p_h;             /* cp_h+cp_l = 2/(3*log2) */
            z_l = cp_l*p_h+p_l*cp+dp_l[k];
        /* log2(ax) = (ss+..)*2/(3*log2) = n + dp_h + z_h + z_l */
            t = (double)n;
            t1 = (((z_h+z_l)+dp_h[k])+t);
            __LO(t1) = 0;
            t2 = z_l-(((t1-t)-dp_h[k])-z_h);
        }

    /* split up y into y1+y2 and compute (y1+y2)*(t1+t2) */
        y1  = y;
        __LO(y1) = 0;
        p_l = (y-y1)*t1+y*t2;
        p_h = y1*t1;
        z = p_l+p_h;
        j = __HI(z);
        i = __LO(z);
        if (j>=0x40900000) {                            /* z >= 1024 */
            if(((j-0x40900000)|i)!=0)                   /* if z > 1024 */
                return s*huge*huge;                     /* overflow */
            else {
                if(p_l+ovt>z-p_h) return s*huge*huge;   /* overflow */
            }
        } else if((j&0x7fffffff)>=0x4090cc00 ) {        /* z <= -1075 */
            if(((j-0xc090cc00)|i)!=0)           /* z < -1075 */
                return s*tiny*tiny;             /* underflow */
            else {
                if(p_l<=z-p_h) return s*tiny*tiny;      /* underflow */
            }
        }
    /*
     * compute 2**(p_h+p_l)
     */
        i = j&0x7fffffff;
        k = (i>>20)-0x3ff;
        n = 0;
        if(i>0x3fe00000) {              /* if |z| > 0.5, set n = [z+0.5] */
            n = j+(0x00100000>>(k+1));
            k = ((n&0x7fffffff)>>20)-0x3ff;     /* new k for n */
            t = zero;
            __HI(t) = (n&~(0x000fffff>>k));
            n = ((n&0x000fffff)|0x00100000)>>(20-k);
            if(j<0) n = -n;
            p_h -= t;
        }
        t = p_l+p_h;
        __LO(t) = 0;
        u = t*lg2_h;
        v = (p_l-(t-p_h))*lg2+t*lg2_l;
        z = u+v;
        w = v-(z-u);
        t  = z*z;
        t1  = z - t*(P1+t*(P2+t*(P3+t*(P4+t*P5))));
        r  = (z*t1)/(t1-two)-(w+z*w);
        z  = one-(r-z);
        j  = __HI(z);
        j += (n<<20);
        if((j>>20)<=0) z = scalbn(z,n); /* subnormal output */
        else __HI(z) += (n<<20);
        return s*z;
}
