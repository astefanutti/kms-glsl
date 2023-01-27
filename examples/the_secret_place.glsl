// Copied from https://www.shadertoy.com/view/7lfXRN
// Created by https://www.shadertoy.com/user/Kamoshika

#define D(v) sin(snoise2D((v) + c * 5e2) * 10.)

float snoise2D(vec2 v);

float hash(float x)
{
    return fract(sin(x) * 43758.5453);
}

vec3 hsv(float h, float s, float v) {
    vec4 t = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(vec3(h) + t.xyz) * 6.0 - vec3(t.w));
    return v * mix(vec3(t.x), clamp(p - vec3(t.x), 0.0, 1.0), s);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 p = (fragCoord * 2. - iResolution.xy) / min(iResolution.x, iResolution.y);
    vec3 col = vec3(0);
    vec3 lightDir = normalize(vec3(-1, 2, 4));
    
    vec2 e = vec2(1e-3, 0);
    
    vec2 q;
    float c, s, L;
    
    for(float i = 0.;i < 20.;i++){
        L = 1. - fract(iTime) + i;
        c = hash(i + ceil(iTime));
        q = p / atan(1e-3, L) / 2e3;
        s = D(q);
        if(s * dot(q, q) > .5){
            break;
        }
    }
    
    vec3 normal = normalize(vec3((-D(q + e.xy) + s)/e.x,
                                 (-D(q + e.yx) + s)/e.x,
                                 1.
                                 ));
    col = hsv(hash(c), .5, 1.) + max(dot(normal, lightDir), 0.);
    L = dot(q, q) * 10. + L * L;
    col *= exp(-L * .01);
    
    fragColor = vec4(col, 1.);
}

/*void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 p = (fragCoord * 2. - iResolution.xy) / min(iResolution.x, iResolution.y);
    vec3 col = vec3(0);
    vec3 lightDir = normalize(vec3(-1, 2, 4));
    
    vec2 e = vec2(1e-4, 0);

    for(float i = 0.;i < 20.;i++){
        float L = 1. - fract(iTime) + i;
        float c = hash(i + ceil(iTime));
        vec2 q = p / atan(1e-3, L) / 2e3;
        float s = D(q);
        if(col.r == 0. && s * dot(q, q) > .5){
            vec3 normal = normalize(vec3((-D(q + e.xy) + s)/e.x,
                                         (-D(q + e.yx) + s)/e.x,
                                         1.
                                         ));
            col = hsv(hash(c), .5, 1.) + max(dot(normal, lightDir), 0.);
            L = dot(q, q) * 20. + L * L;
            col *= exp(-L * .01);
        }
    }
    
    fragColor = vec4(col, 1.);
}*/

//--------------- snoise2D ---------------------------------------------------------------------------
// Description : Array and textureless GLSL 2D simplex noise function.
//      Author : Ian McEwan, Ashima Arts.
//  Maintainer : stegu
//     Lastmod : 20110822 (ijm)
//     License : Copyright (C) 2011 Ashima Arts. All rights reserved.
//               Distributed under the MIT License. See LICENSE file.
//               https://github.com/ashima/webgl-noise
//               https://github.com/stegu/webgl-noise
// 

vec3 mod289(vec3 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec2 mod289(vec2 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec3 permute(vec3 x) {
  return mod289(((x*34.0)+1.0)*x);
}

float snoise2D(vec2 v)
  {
  const vec4 C = vec4(0.211324865405187,  // (3.0-sqrt(3.0))/6.0
                      0.366025403784439,  // 0.5*(sqrt(3.0)-1.0)
                     -0.577350269189626,  // -1.0 + 2.0 * C.x
                      0.024390243902439); // 1.0 / 41.0
// First corner
  vec2 i  = floor(v + dot(v, C.yy) );
  vec2 x0 = v -   i + dot(i, C.xx);

// Other corners
  vec2 i1;
  //i1.x = step( x0.y, x0.x ); // x0.x > x0.y ? 1.0 : 0.0
  //i1.y = 1.0 - i1.x;
  i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  // x0 = x0 - 0.0 + 0.0 * C.xx ;
  // x1 = x0 - i1 + 1.0 * C.xx ;
  // x2 = x0 - 1.0 + 2.0 * C.xx ;
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;

// Permutations
  i = mod289(i); // Avoid truncation effects in permutation
  vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
		+ i.x + vec3(0.0, i1.x, 1.0 ));

  vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
  m = m*m ;
  m = m*m ;

// Gradients: 41 points uniformly over a line, mapped onto a diamond.
// The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)

  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;

// Normalise gradients implicitly by scaling m
// Approximation of: m *= inversesqrt( a0*a0 + h*h );
  m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );

// Compute final noise value at P
  vec3 g;
  g.x  = a0.x  * x0.x  + h.x  * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}
//--------------- snoise2D ---------------------------------------------------------------------------
