// Copied from https://www.shadertoy.com/view/dlSGWm
// Created by https://www.shadertoy.com/user/mrange

// CC0: Warping boxes
//  I tinkered with the multi-level metaballs of yesterday
//  and used boxes instead of circles. This + random tinkering
//  turned out quite nice in the end IMHO.

#define TIME        iTime
#define RESOLUTION  iResolution
#define PI          3.141592654
#define TAU         (2.0*PI)
#define ROT(a)      mat2(cos(a), sin(a), -sin(a), cos(a))

// License: MIT, author: Inigo Quilez, found: https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
float box(vec2 p, vec2 b) {
    vec2 d = abs(p)-b;
    return length(max(d,0.0)) + min(max(d.x,d.y),0.0);
}

// License: MIT OR CC-BY-NC-4.0, author: mercury, found: https://mercury.sexy/hg_sdf/
vec2 mod2(inout vec2 p, vec2 size) {
    vec2 c = floor((p + size*0.5)/size);
    p = mod(p + size*0.5,size) - size*0.5;
    return c;
}

// License: Unknown, author: Hexler, found: Kodelife example Grid
float hash(vec2 uv) {
    return fract(sin(dot(uv, vec2(12.9898, 78.233))) * 43758.5453);
}

// License: Unknown, author: Unknown, found: don't remember
float tanh_approx(float x) {
    //  Found this somewhere on the interwebs
    //  return tanh(x);
    float x2 = x*x;
    return clamp(x*(27.0 + x2)/(27.0+9.0*x2), -1.0, 1.0);
}

float dot2(vec2 p) {
    return dot(p, p);
}

vec2 df(vec2 p, float aa, out float h, out float sc) {
    vec2 pp = p;

    float sz = 2.0;

    float r = 0.0;

    for (int i = 0; i < 5; ++i) {
        vec2 nn = mod2(pp, vec2(sz));
        sz /= 3.0;
        float rr = hash(nn+123.4);
        r += rr;
        if (rr < 0.5) break;
    }

    float d0 = box(pp, vec2(1.45*sz-0.75*aa))-0.05*sz;
    float d1 = sqrt(sqrt(dot2(pp*pp)));
    h = fract(r);
    sc = sz;
    return vec2(d0, d1);
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
    p *= 2.0;
    const mat2 rot0 = ROT(1.0);
    const mat2 rot1 = ROT(-2.0);
    vec2 off0 = 4.0*cos(vec2(1.0, sqrt(0.5))*0.23*TIME);
    vec2 off1 = 3.0*cos(vec2(1.0, sqrt(0.5))*0.13*TIME);
    vec2 sp0 = toSmith(p);
    vec2 sp1 = toSmith((p+off0)*rot0);
    vec2 sp2 = toSmith((p-off1)*rot1);
    vec2 pp = fromSmith(sp0+sp1-sp2);
    p = pp;
    p += 0.25*TIME;

    return p;
}

vec3 effect(vec2 p, vec2 np, vec2 pp) {
    p = transform(p);
    np = transform(np);
    float aa = distance(p, np)*sqrt(2.0);

    float h = 0.0;
    float sc = 0.0;
    vec2 d2 = df(p, aa, h, sc);

    vec3 col = vec3(0.0);

    vec3 rgb = ((2.0/3.0)*(cos(TAU*h+vec3(0.0, 1.0, 2.0))+vec3(1.0))-d2.y/(3.0*sc));
    col = mix(col, rgb, smoothstep(aa, -aa, d2.x));

    const vec3 gcol1 = vec3(.5, 2.0, 3.0);
    col += gcol1*tanh_approx(0.025*aa);

    col = clamp(col, 0.0, 1.0);
    col = sqrt(col);

    return col;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    vec2 q = fragCoord/RESOLUTION.xy;
    vec2 p = -1. + 2. * q;
    vec2 pp = p;
    p.x *= RESOLUTION.x/RESOLUTION.y;
    vec2 np = p+1.0/RESOLUTION.y;
    vec3 col = effect(p, np, pp);
    fragColor = vec4(col, 1.0);
}
