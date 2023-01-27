// Copied from https://www.shadertoy.com/view/dtSGWh
// Created by https://www.shadertoy.com/user/mrange

// CC0: Rainbow smith cells
//  Continuation of earlier experiments

#define TIME        iTime
#define RESOLUTION  iResolution
#define PI          3.141592654
#define PI_2        (0.5*PI)
#define TAU         (2.0*PI)
#define ROT(a)      mat2(cos(a), sin(a), -sin(a), cos(a))

const float rep   = 32.0;
const float over  = 4.0;
const float nstep = 1.0/(rep*over);
const float astep = TAU*nstep;
const float pm    = 17.0;

// License: WTFPL, author: sam hocevar, found: https://stackoverflow.com/a/17897228/418488
const vec4 hsv2rgb_K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
vec3 hsv2rgb(vec3 c) {
    vec3 p = abs(fract(c.xxx + hsv2rgb_K.xyz) * 6.0 - hsv2rgb_K.www);
    return c.z * mix(hsv2rgb_K.xxx, clamp(p - hsv2rgb_K.xxx, 0.0, 1.0), c.y);
}
// License: WTFPL, author: sam hocevar, found: https://stackoverflow.com/a/17897228/418488
//  Macro version of above to enable compile-time constants
#define HSV2RGB(c)  (c.z * mix(hsv2rgb_K.xxx, clamp(abs(fract(c.xxx + hsv2rgb_K.xyz) * 6.0 - hsv2rgb_K.www) - hsv2rgb_K.xxx, 0.0, 1.0), c.y))

// License: MIT OR CC-BY-NC-4.0, author: mercury, found: https://mercury.sexy/hg_sdf/
float modPolar(inout vec2 p, float aa) {
    const float angle = 2.0*PI/rep;
    float a = aa + angle/2.;
    float r = length(p);
    float c = floor(a/angle);
    a = mod(a,angle) - angle/2.;
    p = vec2(cos(a), sin(a))*r;
    // For an odd number of repetitions, fix cell index of the cell in -x direction
    // (cell index would be e.g. -5 and 5 in the two halves of the cell):
    if (abs(c) >= (rep/2.0)) c = abs(c);
    return c;
}

// License: Unknown, author: nmz (twitter: @stormoid), found: https://www.shadertoy.com/view/NdfyRM
vec3 sRGB(vec3 t) {
    return mix(1.055*pow(t, vec3(1./2.4)) - 0.055, 12.92*t, step(t, vec3(0.0031308)));
}

// License: Unknown, author: Matt Taylor (https://github.com/64), found: https://64.github.io/tonemapping/
vec3 aces_approx(vec3 v) {
    v = max(v, 0.0);
    v *= 0.6f;
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return clamp((v*(a*v+b))/(v*(c*v+d)+e), 0.0f, 1.0f);
}

float segmentx(vec2 p, float l, float w) {
    p = abs(p);
    p.x -= l*0.5-w;
    float d0 = length(p)-w;
    float d1 = p.y-w;
    float d = p.x > 0.0 ? d0 : d1;
    return d;
}

vec2 df(vec2 p, float noff, float a, out float n) {
    const float ll  = 0.5;
    const float ss = 0.0015;
    const float bb = ss*4.0;
    n = modPolar(p, a)/rep+noff;
    float m = 16.0*sin(TIME*TAU);
    float anim = sin(TAU*TIME/10.0+pm*noff*TAU);
    p.x -= 0.75+0.25*anim;
    float l = ll*mix(0.5, 1.0, smoothstep(-0.9, 0.9, anim));
    float s = ss;
    float b = bb;
    vec2 p0 = p;
    vec2 p1 = p;
    p1.x = abs(p1.x);
    p1.x -= l*0.5-s;
    float d0 = segmentx(p0, l, s);
    float d1 = length(p1)-b;
    return vec2(d0, d1);
}

// License: Unknown, author: Martijn Steinrucken, found: https://www.youtube.com/watch?v=VmrIDyYiJBA
vec2 hextile(inout vec2 p) {
    // See Art of Code: Hexagonal Tiling Explained!
    // https://www.youtube.com/watch?v=VmrIDyYiJBA
    const vec2 sz       = vec2(1.0, sqrt(3.0));
    const vec2 hsz      = 0.5*sz;

    vec2 p1 = mod(p, sz)-hsz;
    vec2 p2 = mod(p - hsz, sz)-hsz;
    vec2 p3 = dot(p1, p1) < dot(p2, p2) ? p1 : p2;
    vec2 n = ((p3 - p + hsz)/sz);
    p = p3;

    n -= vec2(0.5);
    // Rounding to make hextile 0,0 well behaved
    return floor(n*2.0)*0.5;
}

vec3 effect0(vec2 p, float aa) {
    const float zz = 2.75;
    p /= zz;
    vec2 hn = hextile(p);
    p *= zz;
    float n;
    vec3 col = vec3(0.0);
    const mat2 rr = ROT(TAU/(rep*over));
    vec2 pp = p;
    float a = atan(p.y, p.x);
    float ll = length(p);
    for (float i = 0.0; i < over; ++i) {
        float noff = i*nstep;
        float aoff = i*astep;
        vec2 d = df(p, noff, a-aoff, n);
        d /= aa;

        float g0 = 2.0/max(max(d.x, 0.0), 0.001);
        float g1 = 8.0/max((d.y*d.y), 0.000001);
        col += hsv2rgb(vec3(0.5*ll+n-0.1*TIME, 0.85, g0));
        col += hsv2rgb(vec3(0.5*ll+n-0.1*TIME, 0.5, g1));
        //    col = mix(col, vec3(0.54), smoothstep(1.0, -1.0, d.x));
        //    col = mix(col, vec3(1.0), smoothstep(1.0, -1.0, d.y));
        p *= rr;
    }

    col *= smoothstep(0.5*zz, 0.25*zz, ll);
    const vec3 gcol0 = HSV2RGB(vec3(0.55, 0.75, 10000.0));
    const vec3 gcol1 = HSV2RGB(vec3(0.55, 0.95, 0.025));
    col += gcol0*aa*aa+gcol1/dot(p, p);
    col /= (600.0*aa);
    return col;
}


vec2 toSmith(vec2 p)  {
    // z = (p + 1)/(-p + 1)
    // (x,y) = ((1+x)*(1-x)-y*y,2y)/((1-x)*(1-x) + y*y)
    float d = (1.0 - p.x)*(1.0 - p.x) + p.y*p.y;
    float x = (1.0 + p.x)*(1.0 - p.x) - p.y*p.y;
    float y = 2.0*p.y;
    return vec2(x,y)/d;
}

vec2 fromSmith(vec2 p)  {
    // z = (p - 1)/(p + 1)
    // (x,y) = ((x+1)*(x-1)+y*y,2y)/((x+1)*(x+1) + y*y)
    float d = (p.x + 1.0)*(p.x + 1.0) + p.y*p.y;
    float x = (p.x + 1.0)*(p.x - 1.0) + p.y*p.y;
    float y = 2.0*p.y;
    return vec2(x,y)/d;
}

vec2 transform(vec2 p) {
    vec2 off0 = sin(vec2(1.0, sqrt(0.5))*0.23*TIME);
    vec2 off1 = sin(vec2(1.0, sqrt(0.5))*0.13*TIME);
    vec2 sp0 = toSmith(p);
    vec2 sp1 = toSmith(p+off0);
    vec2 sp2 = toSmith(p-off1);
    vec2 pp = fromSmith(sp0+sp1-sp2);
    pp += 0.1*TIME;
    return pp;
}

vec3 effect(vec2 p, vec2 np, vec2 pp) {
    p = transform(p);
    np = transform(np);
    float aa = distance(p, np)*sqrt(2.0);
    vec3 col = effect0(p, aa);
    return col;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    vec2 q = fragCoord/iResolution.xy;
    vec2 p = -1. + 2. * q;
    vec2 pp = p;
    p.x *= RESOLUTION.x/RESOLUTION.y;
    vec2 np = p + 1.0/RESOLUTION.y;
    vec3 col = effect(p, np, pp);
    col = aces_approx(col);
    col = sRGB(col);
    fragColor = vec4(col, 1.0);
}
