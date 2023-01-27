// Copied from https://www.shadertoy.com/view/7sGSRd
// Created by https://www.shadertoy.com/user/iq

// Copyright Inigo Quilez, 2021 - https://iquilezles.org/
// I am the sole copyright owner of this Work.
// You cannot host, display, distribute or share this Work neither
// as it is or altered, here on Shadertoy or anywhere else, in any
// form including physical and digital. You cannot use this Work in any
// commercial or non-commercial product, website or project. You cannot
// sell this Work and you cannot mint an NFTs of it or train a neural
// network with it without permission. I share this Work for educational
// purposes, and you can link to it, through an URL, proper attribution
// and unmodified screenshot, as part of your educational material. If
// these conditions are too restrictive please contact me and we'll
// definitely work it out.

// This is a figure-8 knot, as described by François Guéritaud, Saul
// Schleimer, and Henry Segerman here:
//
// http://gallery.bridgesmathart.org/exhibitions/2019-icerm-illustrating-mathematics/henrys
//
// It's defined in S3 (ie, the surface of a 4-dimensional hypersphere),
// and then projected to R3 (regular 3D space) through a stereographic
// projection. The extrussion of the path into a tube is done in R3.
// I Tried adaptive subdivision, and worked well, but not completely.

#define AA 1              // make 2 on fast machines

const int   kNum = 256;   // subdivisions. make 1024 on fast machines
const float kRad = 0.06;  // thickness


// knot
vec3 knot( in float t )
{
    t *= 6.283185;

    // knot in S3
    const float e = 0.16;
    const float h = 0.25;
    float a = e*sin(4.0*t);
    float b = 1.0-a*a;
    vec4 q = vec4 (
    b*(h*cos(t)+(1.0-h)*cos(3.0*t)),
    b*(2.0*sqrt(h-h*h)*sin(2.0*t)),
    a*(2.0),
    b*(h*sin(t)-(1.0-h)*sin(3.0*t))
    )
    / (1.0+a*a);

    // rotate in the xw plane (in S3)
    float a1 = iTime*6.283185/10.0;
    q.xw *= mat2(cos(a1),sin(a1),-sin(a1),cos(a1));

    // stereographic projection from S3 to R3
    vec3 p = q.xyz/(1.0-q.w);

    // scale
    return p * 0.25;
}

//-------------------------------------------------------

// intersects a capsule (single cap)
// https://iquilezles.org/articles/intersectors
vec4 iCylinder( in vec3 ro, in vec3 rd,
in vec3 pa, in vec3 pb, float ra,
out float v )
{
    vec4 res = vec4(-1.0);

    v = 0.0;
    vec3 ba = pb-pa;
    vec3 oc = ro-pa;

    float baba = dot(ba,ba);
    float bard = dot(ba,rd);
    float baoc = dot(ba,oc);
    float ocrd = dot(oc,rd);
    float ococ = dot(oc,oc);

    float a = baba - bard*bard;
    float b = baba*ocrd - baoc*bard;
    float c = baba*ococ - baoc*baoc - ra*ra*baba;
    float h = b*b - a*c;
    if( h>0.0 )
    {
        float t = (-b-sqrt(h))/a;

        // body
        float y = baoc + t*bard;
        if( y>0.0 && y<baba )
        {
            v = y/baba;
            res = vec4(t,(oc+t*rd-ba*v)/ra);
        }
        // sphere cap
        else
        {
            h = ocrd*ocrd - ococ + ra*ra;
            if( h>0.0 )
            {
                t = -ocrd - sqrt(h);
                res = vec4(t,(oc+t*rd)/ra);
            }
        }
    }

    return res;
}

// intersects a capsule
// https://iquilezles.org/articles/intersectors
bool sCylinder( in vec3 ro, in vec3 rd,
in vec3 pa, in vec3 pb, float ra )
{
    vec3 ba = pb-pa;
    vec3 oc = ro-pa;

    float baba = dot(ba,ba);
    float bard = dot(ba,rd);
    float baoc = dot(ba,oc);
    float ocrd = dot(oc,rd);
    float ococ = dot(oc,oc);

    float a = baba - bard*bard;
    float b = baba*ocrd - baoc*bard;
    float c = baba*ococ - baoc*baoc - ra*ra*baba;
    float h = b*b - a*c;
    if( h>0.0 )
    {
        float t = (-b-sqrt(h))/a;

        // body
        float y = baoc + t*bard;
        if( t>0.0 && y>0.0 && y<baba ) return true;
        // sphere cap
        h = ocrd*ocrd - ococ + ra*ra;
        if( h>0.0 )
        {
            //if( h*h<-ocrd*abs(ocrd) ) return true;
            t = -ocrd - sqrt(h);
            if( t>0.0 ) return true;
        }
    }

    return false;
}

// intersects the knot
vec4 intersect( in vec3 ro, in vec3 rd, out float resV )
{
    // subdivide the knot, and find intersections
    float   v = 0.0;
    vec4 tnor = vec4(1e20);
    vec3   op = knot(0.0);
    for( int i=1; i<=kNum; i++ )
    {
        // parameter
        float t = float(i)/float(kNum);

        // evaluate knot
        vec3 p = knot(t);

        // segments
        float tmpv;
        vec4 tmp = iCylinder( ro, rd, op, p, kRad, tmpv );
        if( tmp.x>0.0 && tmp.x<tnor.x ) { tnor = tmp; v=t+(tmpv-1.0)/float(kNum); }

        op = p;
    }

    resV = v;

    return tnor;
}

// intersects the knot
float shadow( in vec3 ro, in vec3 rd )
{
    // subdivide the knot, and find intersections
    vec3 op = knot(0.0);
    for( int i=1; i<=kNum; i++ )
    {
        // parameter
        float t = float(i)/float(kNum);

        // evaluate knot
        vec3 p = knot(t);

        // segments
        if( sCylinder( ro, rd, op, p, kRad ) ) return 0.0;

        op = p;
    }

    return 1.0;
}

// do coloring and lighting
vec3 shade( in vec3 pos, in vec3 nor, in vec3 rd, in float hm )
{
    // material - base color
    vec3 mate = 0.5 + 0.5*cos(hm*6.283185+vec3(0.0,2.0,4.0));

    // material - white stripes
    vec3 cen = knot(hm);
    vec3 w = normalize(knot(hm+0.001)-cen);
    vec3 v = vec3(w.y,-w.x,0.0)/length(w.xy);
    vec3 u = normalize(cross(v,w));
    float an = atan( dot(pos-cen,u), dot(pos-cen,v) );
    float ar = an - 30.0*hm + iTime;
    mate += 1.5*smoothstep(-0.3,0.8,sin(6.283185*ar));

    // sky lighting
    vec3 ref = reflect(rd,nor);
    float dif = 0.5+0.5*nor.y;
    float spe = smoothstep(0.1,0.2,ref.y);
    spe *= dif;
    spe *= 0.04 + 0.96*pow( clamp(1.0+dot(rd,nor), 0.0, 1.0), 5.0 );
    if( spe>0.001 ) spe *= shadow(pos+nor*0.001, ref);
    vec3 col = 0.6*mate*vec3(1.0)*dif + spe*6.0;

    // sss
    float fre = clamp(1.0+dot(rd,nor),0.0,1.0);
    col += fre*fre*(0.5+0.5*mate)*(0.2+0.8*dif);

    // self occlusion
    float occ = 0.0;
    for( int i=1; i<=kNum/4; i++ )
    {
        float h = float(i)/float(kNum/4);
        vec3  d = knot(h) - pos;
        float l2 = dot(d,d);
        float l = sqrt(l2);
        float f = dot(d/l,nor);
        occ = max(occ, f*exp2(-l2*8.0) );
        occ = max(occ, f*1.5*kRad*kRad/l2 );
    }
    col *= 1.0-occ;

    return col;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // camera movement
    float an = 0.0*iTime;
    vec3 ro = vec3( 1.0*sin(6.283185*an), 0.0, 1.0*cos(6.283185*an) );
    vec3 ta = vec3( 0.0, 0.02, 0.0 );
    // camera matrix
    vec3 ww = normalize( ta - ro );
    vec3 uu = normalize( cross(ww,vec3(0.0,1.0,0.0) ) );
    vec3 vv = normalize( cross(uu,ww));

    // render
    vec3 tot = vec3(0.0);

    #if AA>1
    for( int m=0; m<AA; m++ )
    for( int n=0; n<AA; n++ )
    {
        // pixel coordinates
        vec2 o = vec2(float(m),float(n)) / float(AA) - 0.5;
        vec2 p = (2.0*(fragCoord+o)-iResolution.xy)/iResolution.y;
        #else
        vec2 p = (2.0*fragCoord-iResolution.xy)/iResolution.y;
        #endif

        // create view ray
        vec3 rd = normalize( p.x*uu + p.y*vv + 1.5*ww );

        // background
        vec3 col = vec3(0.17*(1.0-0.15*dot(p,p))*smoothstep(-1.0,1.0,rd.y));

        // raytrace knot
        float hm;
        vec4 tnor = intersect( ro, rd, hm );
        if( tnor.x<1e19 )
        {
            col = shade( ro+tnor.x*rd, tnor.yzw, rd, hm );
        }

        // gain
        col *= 1.4/(1.0+col);
        // tint
        col = pow( col, vec3(0.8,0.95,1.0) );

        // color to perceptual space
        col = pow( col, vec3(0.4545) );
        tot += col;
        #if AA>1
    }
    tot /= float(AA*AA);
    #endif

    // remove color banding through dithering
    tot += (1.0/255.0)*fract(sin(fragCoord.x*7.0+17.0*fragCoord.y)*1.317);

    fragColor = vec4( tot, 1.0 );
}
