
// Copied from https://www.shadertoy.com/view/NlcGz2
// Created by https://www.shadertoy.com/user/TimelordQ

precision mediump float;

float random(vec2 v) {
    return fract(sin(v.x * 32.1231 - v.y * 2.334 + 13399.2312) * 2412.32312);
}
float random(float x, float y) {
    return fract(sin(x * 32.1231 - y * 2.334 + 13399.2312) * 2412.32312);
}
float random(float x) {
    return fract(sin(x * 32.1231 + 13399.2312) * 2412.32312);
}

float hue2rgb(float f1, float f2, float hue) {
    if (hue < 0.0)
        hue += 1.0;
    else if (hue > 1.0)
        hue -= 1.0;
    float res;
    if ((6.0 * hue) < 1.0)
        res = f1 + (f2 - f1) * 6.0 * hue;
    else if ((2.0 * hue) < 1.0)
        res = f2;
    else if ((3.0 * hue) < 2.0)
        res = f1 + (f2 - f1) * ((2.0 / 3.0) - hue) * 6.0;
    else
        res = f1;
    return res;
}
vec3 hsl2rgb(vec3 hsl) {
    vec3 rgb;
    
    if (hsl.y == 0.0) {
        rgb = vec3(hsl.z); // Luminance
    } else {
        float f2;
        
        if (hsl.z < 0.5)
            f2 = hsl.z * (1.0 + hsl.y);
        else
            f2 = hsl.z + hsl.y - hsl.y * hsl.z;
            
        float f1 = 2.0 * hsl.z - f2;
        
        rgb.r = 0.0; // hue2rgb(f1, f2, hsl.x + (1.0/3.0));
        rgb.g = cos( hue2rgb(f1, f2, hsl.x ));
        rgb.b = 0.0; // hue2rgb(f1, f2, hsl.x - (1.0/3.0));
    }   
    return rgb;
}

float character(float i) {    
     return i<15.01? floor(random(i)*32768.) : 0.;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    vec2 S = 15. * vec2(3., 2.);
    vec2 coord = vec2(
        fragCoord.x / iResolution.y,
        fragCoord.y / iResolution.y + (iResolution.y - iResolution.x) / (9. * iResolution.y)
    );
    vec2 c = floor(coord * S);

    float offset = random(c.x) * S.x;
    float speed = random(c.x * 3.) * 1. + 0.2;
    float len = random(c.x) * 15. + 10.;
    float u = 1. - fract(c.y / len + iTime * speed + offset) * 2.;

    float padding = 2.;
    vec2 smS = vec2(3., 5.);
    vec2 sm = floor(fract(coord * S) * (smS + vec2(padding))) - vec2(padding);
    float symbol = character(floor(random(c + floor(iTime * speed)) * 15.));
    bool s = sm.x < 0. || sm.x > smS.x || sm.y < 0. || sm.y > smS.y ? false
             : mod(floor(symbol / pow(2., sm.x + sm.y * smS.x)), 2.) == 1.;

    vec3 curRGB = hsl2rgb(vec3(c.x / S.x, 1., 0.5));
    if( s )
    {
        if( u > 0.9 )
            {
            curRGB.r = 1.0;
            curRGB.g = 1.0;
            curRGB.b = 1.0;
            }
        else
            curRGB = curRGB * u;
    }
    else
        curRGB = vec3( 0.0, 0.0, 0.0 );

    fragColor = vec4(curRGB.x, curRGB.y, curRGB.z, 1.0);
}
