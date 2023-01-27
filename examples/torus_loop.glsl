// Copied from https://www.shadertoy.com/view/ctlXzN
// Created by https://www.shadertoy.com/user/mrange

// CC0: Torus loop
//  Saw some sweet twitter art again with torus that kept folding in on itself
//  Tried something similar

#define TIME            iTime
#define RESOLUTION      iResolution

#define PI              3.141592654
#define TAU             (2.0*PI)

#define TOLERANCE       0.0001
#define MAX_RAY_LENGTH  22.0
#define MAX_RAY_MARCHES 60
#define NORM_OFF        0.001
#define ROT(a)          mat2(cos(a), sin(a), -sin(a), cos(a))

// License: WTFPL, author: sam hocevar, found: https://stackoverflow.com/a/17897228/418488
const vec4 hsv2rgb_K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
vec3 hsv2rgb(vec3 c) {
    vec3 p = abs(fract(c.xxx + hsv2rgb_K.xyz) * 6.0 - hsv2rgb_K.www);
    return c.z * mix(hsv2rgb_K.xxx, clamp(p - hsv2rgb_K.xxx, 0.0, 1.0), c.y);
}
// License: WTFPL, author: sam hocevar, found: https://stackoverflow.com/a/17897228/418488
//  Macro version of above to enable compile-time constants
#define HSV2RGB(c)  (c.z * mix(hsv2rgb_K.xxx, clamp(abs(fract(c.xxx + hsv2rgb_K.xyz) * 6.0 - hsv2rgb_K.www) - hsv2rgb_K.xxx, 0.0, 1.0), c.y))

const float hoff      = 0.0;
const vec3 skyCol     = HSV2RGB(vec3(hoff+0.57, 0.70, 0.25));
const vec3 glowCol    = HSV2RGB(vec3(hoff+0.025, 0.85, 0.5));
const vec3 sunCol1    = HSV2RGB(vec3(hoff+0.60, 0.50, 0.5));
const vec3 sunCol2    = HSV2RGB(vec3(hoff+0.05, 0.75, 25.0));
const vec3 diffCol    = HSV2RGB(vec3(hoff+0.60, 0.75, 0.25));
const vec3 sunDir1    = normalize(vec3(3., 3.0, -7.0));

mat3  g_rot   ;
float g_anim  ;
mat2  g_rot_yx;
mat2  g_rot_xz;

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

mat3 rot_z(float a) {
    float c = cos(a);
    float s = sin(a);
    return mat3(
    c,s,0
    ,-s,c,0
    , 0,0,1
    );
}

mat3 rot_y(float a) {
    float c = cos(a);
    float s = sin(a);
    return mat3(
    c,0,s
    , 0,1,0
    ,-s,0,c
    );
}

mat3 rot_x(float a) {
    float c = cos(a);
    float s = sin(a);
    return mat3(
    1, 0,0
    , 0, c,s
    , 0,-s,c
    );
}

// License: MIT, author: Inigo Quilez, found: https://iquilezles.org/articles/distfunctions/
float rayPlane(vec3 ro, vec3 rd, vec4 p) {
    return -(dot(ro,p.xyz)+p.w)/dot(rd,p.xyz);
}

// License: MIT, author: Inigo Quilez, found: https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
float box(vec2 p, vec2 b) {
    vec2 d = abs(p)-b;
    return length(max(d,0.0)) + min(max(d.x,d.y),0.0);
}

// License: MIT, author: Inigo Quilez, found: https://iquilezles.org/articles/distfunctions/
float torus(vec3 p, vec2 t) {
    vec2 q = vec2(length(p.xz)-t.x,p.y);
    return length(q)-t.y;
}

float df(vec3 p) {
    const float zz  = 6.0;

    const float r0  = 2.0-0.5;
    const float r1  = r0/zz;
    const float r2  = r1/zz;
    const float r3  = r2/zz;

    float anim  = g_anim;
    float angle = anim*PI;
    float z = mix(zz, 1.0, anim);

    vec3 p0 = p;
    p0 *= g_rot;
    p0.yz *= g_rot_yx;
    p0.x -= mix(r0*zz, 0.0, anim);
    p0 /= z;

    vec3 p1 = p0;

    p1.z = abs(p1.z);
    p1.xz *= g_rot_xz;

    vec3 p2 = p1;
    p2.z = abs(p2.z);
    p2.z -= r0;
    p2 = p2.zxy;

    vec3 p3 = p0;

    float rr = mix(r3, 0.0, anim);
    float d0 = torus(p0, vec2(r0, r1+rr));
    d0 = abs(d0) - r2;
    float d1 = p1.x;
    float d2 = torus(p2, vec2(r1, r2+rr));
    float d3 = p3.x;

    float d = d0;
    d = max(d, d1);
    d = min(d, d2);
    if (angle < PI/4.0) d = max(d, d3);
    if (angle > (TAU-PI/4.0)) d = max(d, -d3);
    return d*z;
}

vec3 normal(vec3 pos) {
    vec2  eps = vec2(NORM_OFF,0.0);
    vec3 nor;
    nor.x = df(pos+eps.xyy) - df(pos-eps.xyy);
    nor.y = df(pos+eps.yxy) - df(pos-eps.yxy);
    nor.z = df(pos+eps.yyx) - df(pos-eps.yyx);
    return normalize(nor);
}

float rayMarch(vec3 ro, vec3 rd, out vec3 gcol) {
    float t = 0.0;
    const float tol = TOLERANCE;
    vec2 dti = vec2(1e10,0.0);
    int i = 0;
    vec3 gc = vec3(0.0);
    for (i = 0; i < MAX_RAY_MARCHES; ++i) {
        float d = df(ro + rd*t);
        gc += (0.0125*glowCol)/d;
        if (d<dti.x) { dti=vec2(d,t); }
        if (d < TOLERANCE || t > MAX_RAY_LENGTH) {
            break;
        }
        t += d;
    }
    gcol = gc;
    if(i==MAX_RAY_MARCHES) { t=dti.y; };
    return t;
}

vec3 render0(vec3 ro, vec3 rd) {
    vec3 col = vec3(0.0);
    float sd = max(dot(sunDir1, rd), 0.0);
    float sf = 1.0001-sd;
    col += clamp(vec3(0.0025/abs(rd.y))*glowCol, 0.0, 1.0);
    col += 0.75*skyCol*pow((1.0-abs(rd.y)), 8.0);
    col += 2.0*sunCol1*pow(sd, 100.0);
    col += sunCol2*pow(sd, 800.0);

    float tp1  = rayPlane(ro, rd, vec4(vec3(0.0, -1.0, 0.0), -6.0));

    if (tp1 > 0.0) {
        vec3 pos  = ro + tp1*rd;
        vec2 pp = pos.xz;
        float db = box(pp, vec2(5.0, 9.0))-3.0;

        col += vec3(4.0)*skyCol*rd.y*rd.y*smoothstep(0.25, 0.0, db);
        col += vec3(0.8)*skyCol*exp(-0.5*max(db, 0.0));
        col += 0.25*sqrt(skyCol)*max(-db, 0.0);
    }

    return clamp(col, 0.0, 10.0);;
}

vec3 render1(vec3 ro, vec3 rd) {
    vec3 gcol;
    int iter;
    float t = rayMarch(ro, rd, gcol);
    vec3 col = render0(ro, rd);

    vec3 p = ro+rd*t;
    vec3 n = normal(p);
    vec3 r = reflect(rd, n);
    float fre = 1.0+dot(rd, n);
    fre *= fre;
    float dif = dot(sunDir1, n);

    if (t < MAX_RAY_LENGTH) {
        col = vec3(0.0);
        col += sunCol1*dif*dif*diffCol*0.25;
        col += mix(0.33, 1.0, fre)*render0(p, r);
    } else {
        col += gcol;
    }

    return col;
}

vec3 effect(vec2 p) {
    float tm  = TIME*0.5+10.0;
    g_rot     = rot_x(0.11*tm)*rot_y(0.23*tm)*rot_z(0.35*tm);
    g_anim    = (0.5+0.5*sin(fract(0.1*TIME)*PI-PI/2.0));
    g_rot_yx  = ROT(g_anim*(PI*0.5));
    g_rot_xz  = ROT(PI/2.0-g_anim*PI);


    vec3 ro = 2.0*vec3(5.0, 1.0, 0.);
    ro.xz *= ROT(-0.1*tm);
    const vec3 la = vec3(0.0, 0.0, 0.0);
    const vec3 up = normalize(vec3(0.0, 1.0, 0.0));

    vec3 ww = normalize(la - ro);
    vec3 uu = normalize(cross(up, ww ));
    vec3 vv = (cross(ww,uu));
    const float fov = tan(TAU/6.);
    vec3 rd = normalize(-p.x*uu + p.y*vv + fov*ww);

    vec3 col = render1(ro, rd);

    return col;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    vec2 q = fragCoord/iResolution.xy;
    vec2 p = -1. + 2. * q;
    vec2 pp = p;
    p.x *= RESOLUTION.x/RESOLUTION.y;
    vec3 col = vec3(0.0);
    col = effect(p);
    col *= smoothstep(1.5, 0.5, length(pp));
    col = aces_approx(col);
    col = sRGB(col);
    fragColor = vec4(col, 1.0);
}
