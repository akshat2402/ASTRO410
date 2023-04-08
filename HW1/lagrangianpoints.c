/*
 ** lagrangianpoints.c -- solution to Homework1 Q1
 **
 ** UNITS: G = 1
 */

#include <stdio.h>
#include <stdlib.h> /* for atof() */
#include <math.h>
#include <assert.h>

#include "/Users/akshatchaturvedi/Documents/PennState/Spring2023/Astro410/rtbis.c"

#define GRID_FILE "lagrange_grid.dat"
#define ROOT_FILE "lagrange_root.dat"

/* grid dimensions and spacing */

#define XMIN (-2.0)
#define XMAX   2.0
#define DX     0.1

#define YMIN (-2.0)
#define YMAX   2.0
#define DY     0.1

#define FEPS 1.0e-6 /* floating point accuracy */

#define SQ(x) ((x)*(x))

float m1,m2,w2,x1,x2,xc; /* global declarations needed for some functions */

float gravx(float ir13,float ir23,float x,float y)
{
    /*
     ** Returns x component of effective gravity at point (x,y).
     ** ir13 and ir23 must be precomputed: these are the inverse cubes
     ** of the separations between x1 & x and x2 & x, respectively.
     ** Note m1, m2, x1, x2, and w2 are global variables.
     */

    return - m1*(x - x1)*ir13 - m2*(x - x2)*ir23 + w2*x;
    }

float gravy(float ir13,float ir23,float x,float y)
{
    /*
     ** Returns y component of effective gravity at point (x,y).
     ** See gravx() for more detail.
     */

    return - m1*y*ir13 - m2*y*ir23 + w2*y;
    }

float fx(float x)
{
    /*
     ** Specialized function for rtbis(): computes effective gravity
     ** at any point on x axis (y=0). For finding L1, L2, & L3.
     */

    double r12 = SQ(x - x1);
    double r22 = SQ(x - x2);

    assert(r12 > 0.0 && r22 > 0.0);
    return gravx(1.0/(r12*sqrt(r12)),1.0/(r22*sqrt(r22)),x,0.0);
    }

float fy(float y)
{
    /*
     ** Specialized function for rtbis(): computes effective gravity
     ** at any point on x=xc axis. For finding L4 & L5.
     */

    double r12 = SQ(xc - x1) + SQ(y);
    double r22 = SQ(xc - x2) + SQ(y);

    assert(r12 > 0.0 && r22 > 0.0);
    return gravy(1.0/(r12*sqrt(r12)),1.0/(r22*sqrt(r22)),xc,y);
    }

int main(int argc,char *argv[])
{
    FILE *fp;
    double x,y; /* doubles to reduce roundoff */
    float d,m,r2,r12,r22,isqrtr12,isqrtr22,ir13,ir23;
    float p,gx,gy,lp;

    if (argc != 4) {
        (void) fprintf(stderr,"Usage: %s m_1 m_2 sep\n",argv[0]);
        return 1;
        }

    if ((m1 = atof(argv[1])) <= 0.0 || (m2 = atof(argv[2])) <= 0.0) {
        (void) fprintf(stderr,"Masses must be positive\n");
        return 1;
        }

    if ((d = atof(argv[3])) <= 0.0) {
        (void) fprintf(stderr,"Separation must be positive\n");
        return 1;
        }

    if (d > XMAX - XMIN || d > YMAX - YMIN) {
        (void) fprintf(stderr,"Separation exceeds grid dimensions\n");
        return 1;
        }

    if (d > 0.4*(XMAX - XMIN) || d > 0.4*(YMAX - YMIN))
        (void) fprintf(stderr,"WARNING: large separation\n");

    m = m1 + m2; /* total mass */

    w2 = m/(d*d*d); /* square of orbital frequency */

    /*
     ** Compute locations of masses along x axis. Strategy: the center
     ** of mass will be at the origin of our rotating frame.
     */

    x1 = - m2*d/m; /* arbitrarily take x1 negative */
    x2 =   m1*d/m;

    /* helpful fact: L4 & L5 lie on vertical axis midway between x1 & x2 */

    xc = 0.5*(x1 + x2);

    /* prepare output file */

    if ((fp = fopen(GRID_FILE,"w")) == NULL) {
        (void) fprintf(stderr,"Unable to open \"%s\" for writing\n",GRID_FILE);
        return 1;
        }

    /* compute gravity and potential at grid points */

    (void) printf("Writing grid data to \"%s\"...\n",GRID_FILE);

    /* format: x y potential grav-x grav-y */

    for (x=XMIN;x<=XMAX+FEPS;x+=DX) /* +FEPS to avoid roundoff */
        for (y=YMIN;y<=YMAX+FEPS;y+=DY) {
            r2 = SQ(x) + SQ(y);
            r12 = SQ(x - x1) + SQ(y); /* recall y1, y2 == 0 */
            r22 = SQ(x - x2) + SQ(y);
            r12 = (r12 < FEPS ? FEPS : r12); /* avoid singularities */
            r22 = (r22 < FEPS ? FEPS : r22);
            isqrtr12 = 1.0/sqrt(r12);
            isqrtr22 = 1.0/sqrt(r22);
            ir13 = isqrtr12/r12;
            ir23 = isqrtr22/r22;
            gx = gravx(ir13,ir23,x,y);
            gy = gravy(ir13,ir23,x,y);
            p = - m1*isqrtr12 - m2*isqrtr22 - 0.5*w2*r2;
            (void) fprintf(fp,"%e %e %e %e %e\n",x,y,p,gx,gy);
            }

    (void) fclose(fp);

    /* now find Lagrange points */

    if ((fp = fopen(ROOT_FILE,"w")) == NULL) {
        (void) fprintf(stderr,"Unable to open \"%s\" for writing\n",ROOT_FILE);
        return 1;
        }

    (void) printf("Writing Lagrange points to \"%s\"...\n",ROOT_FILE);

    /* L1: on x axis, between x1 and x2 */

    lp = rtbis(fx,x1 + FEPS,x2 - FEPS,FEPS);
    (void) fprintf(fp,"%g 0\n",lp);

    /* L2: on x axis, to the right of x2 */

    lp = rtbis(fx,x2 + FEPS,XMAX,FEPS);
    (void) fprintf(fp,"%g 0\n",lp);

    /* L3: on x axis, to the left of x1 */

    lp = rtbis(fx,XMIN,x1 - FEPS,FEPS);
    (void) fprintf(fp,"%g 0\n",lp);

    /* L4: on positive x=xc axis */

    lp = rtbis(fy,FEPS,YMAX,FEPS);
    (void) fprintf(fp,"%g %g\n",xc,lp);

    /* L5: on negative x=xc axis */

    lp = rtbis(fy,YMIN,-FEPS,FEPS);
    (void) fprintf(fp,"%g %g\n",xc,lp);

    (void) fclose(fp);

    return 0;
    }
