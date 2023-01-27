// Copied from https://www.shadertoy.com/view/stfXWM
// Created by https://www.shadertoy.com/user/Taron

// "Gyroidal Morphoma" 
// Based on Martijn Steinrucken aka The Art of Code/BigWings - 2020
// https://www.shadertoy.com/view/WtKSRz
//
// I feel a little funny, almost just adjusting a shader, but this one's too much fun!
// I've added some simple gloss and reflection, besides the organic wobbles.
// If you want to truly learn something, check out his amazing tutorials on youtube (The Art of Code)!
// And, Martijn, if you read this: I'm a fan and bow before your excellence, especially your teaching style.
// Technically I may know much of it for almost 30 years, but the finesse and virtuosity of you is a pleasure 
// to watch and utterly inspiring!

#define MAX_STEPS 100
#define MAX_DIST 100.
#define SURF_DIST .001

#define S smoothstep
#define T iTime

mat2 Rot(float a) {
    float s=sin(a), c=cos(a);
    return mat2(c, -s, s, c);
}

float sdBox(vec3 p, vec3 s) {
    p = abs(p)-s;
	return length(max(p, 0.))+min(max(p.x, max(p.y, p.z)), 0.);
}

float Gyroid(vec3 p, float offset, float scale){
    p *=scale;
    offset +=.025*p.y;
    return (dot(sin(p),cos(p.zxy))+offset)/scale;
}

float getGyroids(vec3 p){
    p.z -=iTime*.25;
    float gyroid = Gyroid(p,1.2,10.);
    gyroid -= 0.5*Gyroid(p+vec3(0.15,0.,-0.05),1.2,19.79);
    gyroid += 0.25*Gyroid(p+vec3(7.15+iTime*.1,0.,-0.05*gyroid),1.,29.39);
    gyroid += 0.125*Gyroid(p+vec3(-2.15,0.3-gyroid,0.05),.9,49.99);
    gyroid += 0.065*Gyroid(p+vec3(-0.05,0.1,7.15+gyroid),0.95,79.99);
    return gyroid;

}

float GetDist(vec3 p) {
    float box = sdBox(p, vec3(2.));
    p.xy *= Rot(p.z*.73);
    float gyroid = getGyroids(p);
    
    
    float d = max(gyroid*0.6, box);
    return d;
}

float RayMarch(vec3 ro, vec3 rd) {
	float dO=0.;
    
    for(int i=0; i<MAX_STEPS; i++) {
    	vec3 p = ro + rd*dO;
        float dS = GetDist(p);
        dO += dS;
        if(dO>MAX_DIST || abs(dS)<SURF_DIST) break;
    }
    
    return dO;
}

vec3 GetNormal(vec3 p) {
	float d = GetDist(p);
    vec2 e = vec2(.001, 0);
    
    vec3 n = d - vec3(
        GetDist(p-e.xyy),
        GetDist(p-e.yxy),
        GetDist(p-e.yyx));
    
    return normalize(n);
}

vec3 GetRayDir(vec2 uv, vec3 p, vec3 l, float z) {
    vec3 f = normalize(l-p),
        r = normalize(cross(vec3(0,1,0), f)),
        u = cross(f,r),
        c = f*z,
        i = c + uv.x*r + uv.y*u,
        d = normalize(i);
    return d;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = (fragCoord-.5*iResolution.xy)/iResolution.y;
	vec2 m = iMouse.xy/iResolution.xy;

    vec3 ro = vec3(0., 0., -1.);
    ro.yz *= Rot(-m.y*3.14+0.15);
    ro.xz *= Rot(-m.x*6.2831-3.15);
    
    vec3 rd = GetRayDir(uv, ro, vec3(0,0.,0), 1.);
    vec3 col = vec3(0);
   
    float d = RayMarch(ro, rd);

    if(d<MAX_DIST) {
        vec3 p = ro + rd * d;
        vec3 n = GetNormal(p);
        float dif = n.y*.5+.5;
        col = vec3(dif);
        
        // gloss and pseudo reflections below 0.5y
        if(p.y<0.5){
            vec3 r = reflect(rd, n);
            float refl = 0.;
            float m = RayMarch(p-n*.005, r);
            if(m<MAX_DIST){
                vec3 rn = GetNormal(ro+r*m);
                refl = rn.y*.5+.5;
            }

            vec3 spec = pow(max(0.,dot(vec3(0.,1.,0.),r)),53.3)*.5+min(1.,refl)*vec3(0.02,0.15,0.21);
            col +=spec *max(0.,min(1.,0.5-p.y*2.5));
        }
     }
    
    col = pow(col, vec3(0.4545,1.0545,1.4545));	// colored gamma correction
    col = mix(col, mix(vec3(0.1,0.15,0.3),vec3(0.1,0.15,0.3)*10.,rd.y*.3),min(1.0,d*.5)); // fog
    col = mix(col, vec3(0.24,0.03,0.02),length(uv)); // vignette
    
    fragColor = vec4(col,1.0);
}
